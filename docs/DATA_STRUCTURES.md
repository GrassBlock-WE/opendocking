# OpenDocking data structures

This is the **step 2 deliverable** of the project brief: the core Rust data
structures of `dock-core`, with their real definitions, a field-by-field
commentary, the invariant each one maintains, and the place in AutoDock Vina
that the equivalent lives in.

Every definition below is taken from the source in this repository; nothing is
paraphrased into a shape the code does not have. Private fields are marked as
such and are shown for completeness. The upstream column names the class, struct
or function in AutoDock Vina (the upstream C++ sources, Apache-2.0,
Copyright (c) 2006–2010, The Scripps Research Institute) that the structure
mirrors; the AutoDock 4 force field (GPL) is the reference for the AD4 typing
and terms. The preparation layer follows the *behaviour* documented by Meeko
(LGPL-2.1) but shares no code with it.

---

## Conventions

| Quantity | Unit / convention |
|---|---|
| Distances, coordinates | Å (Ångström) |
| Energies | kcal/mol |
| Angles | radians, wrapped into `(-pi, pi]` |
| Quaternion layout | `(x, y, z, w)`, `w` scalar, unit length |
| Rotation matrix | matches `glam::DMat3::from_quat` and Vina's `quaternion_to_r3` |
| Atom indices | `usize`, `0`-based, into the owning container |
| Discrimination sentinel | `XsType::W` means "no X-Score type"; `AdType::Unknown` means "no AD4 type" |

Two typing schemes coexist, exactly as in AutoDock Vina:

* **AD4 typing** (`AdType`) is what the PDBQT file carries in its last two
  columns. It is required to parse a PDBQT file and to evaluate the AD4.2 force
  field.
* **X-Score typing** (`XsType`) is what the Vina and Vinardo force fields use.
  It is **re-derived from the element, the AD4 type and the bonding graph** —
  never read from the file — so that a hydrogen can turn its heavy neighbour
  into an H-bond donor.

The invariants that hold across the whole crate:

1. `Atom::ad`, `Atom::el` and `Atom::xs` are mutually consistent; `assign_types`
   re-derives `el` and `xs` from `ad` plus the bond graph, and is idempotent.
2. `Bond::other` always indexes an atom of the *same* container, and every bond
   list is symmetric and sorted by `other`.
3. `Pair::a < Pair::b` for every pair produced by the crate.
4. Atom indices inside `KinematicTree::local`, `MovableModel::atoms`,
   `MovableModel::coords` and `MovableModel::minus_forces` all use the same flat
   movable-atom numbering that `dock_core::io::pdbqt` assigns while flattening
   the PDBQT topology.
5. Every half-open atom range `(begin, end)` is a valid slice of that flat
   numbering, and the ranges of the frames of one tree are disjoint and
   exhaust the tree's own range.
6. A `Conf` is dimensionally consistent with the `DofLayout` it is used with.
7. `AffinityGrid::data.len() == shape[0] * shape[1] * shape[2] * types.len()`.

---

## Atom typing

### `Element`

```rust
// crates/dock-core/src/atom.rs
/// Chemical element as far as the force fields care.
///
/// Discriminants follow Vina's `EL_TYPE_*` constants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum Element {
    H = 0,
    C = 1,
    N = 2,
    O = 3,
    S = 4,
    P = 5,
    F = 6,
    Cl = 7,
    Br = 8,
    I = 9,
    Si = 10,
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
    pub fn from_symbol(s: &str) -> Option<Element>;

    /// `true` for elements that are not C and not H.
    pub fn is_heteroatom(self) -> bool;

    /// Single-letter-ish symbol for output.
    pub fn symbol(self) -> &'static str;
}
```

**Field by field.** This is a `repr(u8)` enum, so the discriminant *is* the
storage. `from_symbol` strips trailing digits (so the PDB atom name `CA` — alpha
carbon — parses as carbon, while `CL` parses as chlorine), accepts `D` as
hydrogen, and returns `None` for anything it does not know. `Met` collapses
every metal: the AD4 table has separate types for Mg/Mn/Zn/Ca/Fe, but the
X-Score scheme does not. `Dummy` covers macrocycle closure pseudo-atoms.
`Size` is the "beyond the end" sentinel and is never produced by typing.

**Invariant.** `Element::COUNT == 15` and the discriminants are contiguous from
`H = 0` to `Size = 14`; `AdType::element()` and `XsType::from_element()` both
switch on the discriminants, so they must never be reordered.

**Vina equivalent.** `atom_type::el` plus the `EL_TYPE_*` constants in
`atom_constants.h` (Vina's element enumeration is `EL_TYPE_H, EL_TYPE_C, ...,
EL_TYPE_SIZE`). OpenDocking adds the `from_symbol` parser, which Vina keeps in
its PDBQT reader.

---

### `AdType`

```rust
// crates/dock-core/src/atom.rs
/// AutoDock 4 atom type (the PDBQT atom-type column).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum AdType {
    C = 0,      // aliphatic carbon, PDBQT "C"
    A = 1,      // aromatic carbon, PDBQT "A"
    N = 2,      // nitrogen
    O = 3,      // oxygen
    P = 4,
    S = 5,
    H = 6,      // non-polar hydrogen
    F = 7,
    I = 8,
    NA = 9,     // H-bond-accepting nitrogen
    OA = 10,    // H-bond-accepting oxygen
    SA = 11,    // H-bond-accepting sulfur
    HD = 12,    // polar hydrogen
    Mg = 13, Mn = 14, Zn = 15, Ca = 16, Fe = 17,
    Cl = 18, Br = 19, Si = 20, At = 21,
    G0 = 22, G1 = 23, G2 = 24, G3 = 25,   // macrocycle closure dummies
    Cg0 = 26, Cg1 = 27, Cg2 = 28, Cg3 = 29, // closure carbons
    W = 30,     // hydrated-ligand pseudo atom
    /// Unknown / unsupported (Vina's `AD_TYPE_SIZE`).
    Unknown = 31,
}

impl AdType {
    /// Number of *known* AD4 types (excludes `Unknown`).
    pub const COUNT: usize = 31;

    /// All known types, in table order.
    pub const ALL: [AdType; 31];

    /// Discriminant as `usize`, clamped to the `Unknown` sentinel.
    pub fn index(self) -> usize;

    /// Build from a raw discriminant.
    pub fn from_index(i: usize) -> AdType;

    /// Two-letter PDBQT name.
    pub fn name(self) -> &'static str;

    /// Parse the PDBQT atom-type column.
    pub fn from_name(name: &str) -> AdType;

    /// Element implied by this AD4 type.
    pub fn element(self) -> Element;

    pub fn is_hydrogen(self) -> bool;      // H | HD
    pub fn is_heteroatom(self) -> bool;    // Vina's `ad_is_heteroatom`
    pub fn is_glue(self) -> bool;          // G0..G3
}
```

Companion tables, all indexed by the discriminant:

```rust
// crates/dock-core/src/atom.rs
/// `atom_kind_data[].name`, in table order (Vina `atom_constants.h`).
pub const ATOM_KIND_NAMES: [&str; 31] = [
    "C", "A", "N", "O", "P", "S", "H", "F", "I", "NA", "OA", "SA", "HD", "Mg",
    "Mn", "Zn", "Ca", "Fe", "Cl", "Br", "Si", "At", "G0", "G1", "G2", "G3",
    "CG0", "CG1", "CG2", "CG3", "W",
];

/// AutoDock 4 atom-kind properties, ported from `atom_kind_data[]`.
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

/// The AD4 parameter table (31 entries, index == `AdType` discriminant).
pub const ATOM_KIND_DATA: [AtomKind; 31] = [ /* ... */ ];

/// Properties of an AD4 type (clamped for the `Unknown` sentinel).
pub fn ad_type_property(t: AdType) -> AtomKind;

/// Solvation parameter assigned to metals without an AD4 type.
pub const METAL_SOLVATION_PARAMETER: f64 = -0.00110;

/// Largest covalent radius in the AD4 table (Vina's `max_covalent_radius`).
pub fn max_covalent_radius() -> f64;

/// Covalent radius for an arbitrary AD4 type, falling back to the table maximum
/// outside the table (Vina's `assign_bonds` behaviour).
pub fn covalent_radius(t: AdType) -> f64;
```

**Field by field.** `AtomKind::radius` and `depth` feed the AD4 12-6 term;
`hb_depth` *is the sign carrier*: a pair is an H-bond exactly when the product of
the two `hb_depth` values is negative (acceptors are `-5.0` for `NA`/`OA`, `-1.0`
for `SA`; donors are `+1.0` for `HD`), which is why the 12-6 and the 12-10 terms
are mutually exclusive per pair. `hb_radius` is the 12-10 optimum.
`solvation`/`volume` feed the desolvation term, `covalent_radius` feeds bond
perception and the "optimal bond length" test.

Notable table values (all from Vina's `atom_kind_data[]`, which was generated
from `AD4_parameters.data`): `C` radius 2.00 Å, depth 0.15, covalent radius
0.77; `OA` radius 1.60, depth 0.20, `hb_depth` −5.0, `hb_radius` 1.9; `HD` radius
1.00, `hb_depth` +1.0, `hb_radius` 0.0; closure dummies `G0`..`G3` and `W` are
all zero; the maximum covalent radius in the table is calcium at 1.74 Å.

**Invariants.**

* `ATOM_KIND_DATA.len() == ATOM_KIND_NAMES.len() == AdType::COUNT == 31`, and
  `ATOM_KIND_NAMES[i]` round-trips through `AdType::from_name`.
* The discriminants are part of the PDBQT wire format and of the table layout:
  **never reorder them.**
* `from_name` reproduces Vina's `string_to_ad_type`, including `Se -> S` (its
  `atom_equivalence_data`) and the non-AD metal list
  (`Cu, Fe, Na, K, Hg, Co, U, Cd, Ni`) which maps to `Unknown`, and case-folds
  the bare symbols `MG/MN/ZN/CA`.
* `ad_type_property` clamps out-of-range discriminants to the last table entry
  (`W`), so it never panics even for a hypothetical future type.

**Vina equivalent.** `atom_type` in `atom_type.h` (the `t` field plus
`ad_type`), `atom_constants.h`'s `atom_kind_data[]`, `AD_TYPE_SIZE`, the
`string_to_ad_type` function and `max_covalent_radius`. The `AtomKind` struct
mirrors the `atom_kind_data` record exactly.

---

### `XsType`

```rust
// crates/dock-core/src/atom.rs
/// X-Score atom type used by the Vina and Vinardo force fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum XsType {
    CH = 0, CP = 1,            // hydrophobic / polar carbon
    NP = 2, ND = 3, NA = 4, NDA = 5,
    OP = 6, OD = 7, OA = 8, ODA = 9,
    SP = 10, PP = 11,
    FH = 12, ClH = 13, BrH = 14, IH = 15,
    Si = 16, At = 17, MetD = 18,
    CHCg0 = 19, CPCg0 = 20, G0 = 21,
    CHCg1 = 22, CPCg1 = 23, G1 = 24,
    CHCg2 = 25, CPCg2 = 26, G2 = 27,
    CHCg3 = 28, CPCg3 = 29, G3 = 30,
    /// Hydrated-ligand pseudo atom; no X-Score type (the sentinel).
    W = 31,
}

impl XsType {
    pub const COUNT: usize = 32;
    pub fn index(self) -> usize;
    pub fn from_index(i: usize) -> XsType;   // clamps to `W`
    pub fn radius(self) -> f64;              // Vina radii table
    pub fn vinardo_radius(self) -> f64;      // Vinardo radii table
    pub fn is_hydrophobic(self) -> bool;     // CH | FH | ClH | BrH | IH
    pub fn is_acceptor(self) -> bool;        // NA | NDA | OA | ODA
    pub fn is_donor(self) -> bool;           // ND | NDA | OD | ODA | MetD
    pub fn h_bond_possible(self, other: XsType) -> bool;
    pub fn is_glue(self) -> bool;            // G0..G3

    /// Compute the X-Score type from the element, the AD4 type and the
    /// *bonding graph* (ported from Vina's `model::assign_types`).
    pub fn from_element(
        el: Element,
        ad: AdType,
        bonded_to_hd: bool,
        bonded_to_heteroatom: bool,
    ) -> XsType;
}

/// All X-Score types in discriminant order.
pub const XS_ALL: [XsType; 32];

/// Vina force-field van der Waals radii, `xs_vdw_radii[]`.
pub const XS_VDW_RADII: [f64; 32];

/// Vinardo force-field van der Waals radii, `xs_vinardo_vdw_radii[]`.
pub const XS_VINARDO_VDW_RADII: [f64; 32];

/// Vina's `optimal_distance`: the sum of X-Score radii, or `0` for glue types.
pub fn optimal_distance(t1: XsType, t2: XsType) -> f64;

/// Vina's `optimal_distance_vinardo`.
pub fn optimal_distance_vinardo(t1: XsType, t2: XsType) -> f64;

/// Vina's `is_glued`: which (closure dummy, closure carbon) pairs are glued.
pub fn is_glued(t1: XsType, t2: XsType) -> bool;
```

**Field by field / semantics.** The donor/acceptor flags are the whole point of
the X-Score scheme: they are *derived*, not stored. `from_element` computes

```text
acceptor = (ad == OA) || (ad == NA)      // from the file's AD4 type
donor    = (el == Met) || bonded_to_hd    // from the *bonding graph*
```

and then picks `ND/NA/NDA/NP` for nitrogen and `OD/OA/ODA/OP` for oxygen from the
`(acceptor, donor)` pair. Carbon becomes `CP`/`CH` (or a closure-carbon variant)
depending on `bonded_to_heteroatom`; sulfur becomes `SP`, phosphorus `PP`,
halogens `FH/ClH/BrH/IH`, metals `MetD`. Any hydrogen becomes `W` — see the
curation rule in [`SCORING.md`](SCORING.md#hydrogens-and-x-score-typing).

`optimal_distance` returns the sum of the two Vina radii, and **exactly 0.0**
when either partner is a closure dummy, so the reduced distance of a glue pair
equals the raw distance. `is_glued` enumerates the sixteen legal
(dummy, closure-carbon) combinations; a dummy paired with anything else is an
"unmatched closure dummy" and is excluded from the pair list altogether.

**Invariants.**

* `XS_ALL[t.index()] == t`, `XsType::COUNT == 32`, and the radii tables have
  exactly 32 entries in discriminant order (pinned by the
  `xs_radii_are_ordered_with_the_enum` test: `CH = 1.9`, `MetD = 1.2`,
  `G0 = G2 = 1.9`, `W = 0.0`; Vinardo `CH = 2.0`, `OD = 1.6`, `W = 0.0`).
* `from_index` clamps invalid values to `W`, which is Vina's "no W atoms in XS
  types" mapping of `XS_TYPE_SIZE`.
* `h_bond_possible` is symmetric by construction, and `ND`/`ND` is *not* an
  H-bond (pinned by `h_bond_flags_are_symmetric`).
* **Never reorder the discriminants**: the radii tables, `GRID_TYPES`, the
  `slot` array of `AffinityGrid`, the WGSL shader's closure-dummy test and the
  Python `odock.XS_TYPES` list all index by them.

**Vina equivalent.** `atom_type::xs` in `atom_type.h`, `xs_vdw_radii[]` /
`xs_vinardo_vdw_radii[]` / `XS_TYPE_SIZE` in `atom_constants.h`,
`optimal_distance` / `optimal_distance_vinardo` / `xs_is_hydrophobic` /
`xs_is_acceptor` / `xs_is_donor` / `is_glued` in `atom_constants.h`, and
`model::assign_types` in `model.cpp` for the graph-derived typing.

---

## The molecular graph

### `Atom`

```rust
// crates/dock-core/src/molecule.rs
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
    /// Create an atom with no bonds; typing is filled in by `perceive_bonds`.
    pub fn new(coords: DVec3, ad: AdType, charge: f64) -> Atom;

    pub fn is_hydrogen(&self) -> bool;                 // el == Element::H
    pub fn is_heteroatom(&self) -> bool;               // ad.is_heteroatom()
    pub fn optimal_covalent_bond_length(&self, other: &Atom) -> f64;
}
```

**Field by field.**

* `serial` / `name` / `res_name` / `chain_id` / `res_id` are bookkeeping: they
  are copied out of the PDBQT line so the writer can reproduce it, and they are
  never used by the numerics.
* `coords` is the *current* laboratory-frame position. For movable atoms it is
  rewritten by every forward-kinematics pass; for receptor atoms it never
  changes.
* `ad` is authoritative for the AD4 force field and for parsing; `el` and `xs`
  are derived from it (plus `bonds` for `xs`). `charge` is read from PDBQT
  columns 71–76 and is used only by the AD4 electrostatics (`332 * q1 * q2 /
  (r * eps(r))`); Vina and Vinardo ignore it entirely.
* `type_name` keeps the raw file token, which matters when a file uses a
  non-canonical spelling that the writer must preserve.
* `bonds` is the perceived adjacency list (see `Bond`). It is empty until
  `perceive_bonds` runs.

**Invariants.**

* `el == ad.element()` and `xs == XsType::from_element(el, ad, bonded_to_hd,
  bonded_to_heteroatom)` after `assign_types`; `Atom::new` seeds `xs` with the
  graph-free guess `from_element(el, ad, false, false)`, so a freshly built atom
  is *provisional* until perception runs.
* `bonds` is sorted by `Bond::other` and symmetric: `bonds` contains `j` iff
  `atoms[j].bonds` contains `i`.
* `coords` is finite; the scoring code assumes no `NaN` (the energy container
  filters non-finite poses at the end of `Docking::run`).

**Vina equivalent.** The three-level hierarchy `atom_type` → `atom_base` →
`atom` in `atom_type.h`, `atom_base.h`, `atom.h`. `coords` corresponds to
Vina's parallel `vecv` coordinate array rather than to a per-atom field;
OpenDocking keeps the position inside the atom for the *template* and inside
`MovableModel::coords` for the *live* state. `serial`, `name`, `res_name`,
`chain_id` and `res_id` belong to Vina's `parsed_line`/context machinery, and
`type_name` is Vina's saved atom-type string.

---

### `Bond`

```rust
// crates/dock-core/src/molecule.rs
/// A perceived covalent bond.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    /// Index of the partner atom inside the same `Molecule`.
    pub other: usize,
    /// Inter-atomic distance at perception time (Å).
    pub length: f64,
    /// Whether this bond is a rotatable torsion in the input topology.
    pub rotatable: bool,
}

/// Bond-length tolerance factor (Vina's `bond_length_allowance_factor`).
pub const BOND_LENGTH_ALLOWANCE_FACTOR: f64 = 1.1;
```

**Field by field.** `other` is a *half-edge* target: the bond is stored once on
each endpoint. `length` is the distance measured when the bond was perceived; it
is the reference length that the `.map`/typing routines and the tests use, and
it is not updated by kinematics (an isometry preserves it). `rotatable` is the
flag Vina carries in the same struct; OpenDocking's geometric perception always
writes `false` and the rotatable set is taken from the PDBQT `BRANCH` topology
instead (`ParsedLigand::rotors`), because that is what the user's preparation
decided.

**Invariants.** `other != self index`; the pair is symmetric; `length > 0`.
`rotatable` is informational for the kernel's own perception path and is not
consulted by the search.

**Vina equivalent.** `bond` in `atom.h` (`atom_index connected_atom_index; fl
length; bool rotatable;`) — a direct correspondence, with the index named
`connected_atom_index` upstream. Vina's `assign_bonds` in `model.cpp` is the
reference for the distance test and the "third atom in the lens" rejection.

---

### `Molecule`

```rust
// crates/dock-core/src/molecule.rs
/// A molecule: a flat atom list plus perceived connectivity.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Molecule {
    /// The atoms.
    pub atoms: Vec<Atom>,
}

impl Molecule {
    pub fn new() -> Molecule;
    pub fn from_atoms(atoms: Vec<Atom>) -> Molecule;
    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;
    /// Arithmetic mean of all atom positions (Vina's `model::center`).
    pub fn center(&self) -> DVec3;
    /// Bounding box as `(min, max)`.
    pub fn bounding_box(&self) -> (DVec3, DVec3);
    /// Look up an atom index by PDBQT serial number.
    pub fn index_of_serial(&self, serial: i32) -> Option<usize>;
    /// Perceive bonds and (re)assign X-Score atom types.
    pub fn perceive(&mut self);
    /// Append `other`, shifting bond indices, then re-perceive.
    pub fn append(&mut self, other: &Molecule);
    /// Remove every atom for which `keep` returns `false`, and remap bonds.
    pub fn retain<F: Fn(&Atom) -> bool>(&mut self, keep: F);
}
```

**Field by field.** There is exactly one field: a flat atom vector. All the
structure — rings, fragments, the rigidity pattern — is *derived*. `perceive` is
the two-pass pipeline `perceive_bonds` then `assign_types`; `retain` drops
atoms, drops bonds that pointed at them and re-indexes the survivors;
`append` shifts the incoming bond indices by the current length (it does not
re-perceive by itself — the doc comment says so, and the caller does).

**Invariants.** Every bond index is in range for `atoms`. `retain` breaks the
"bonds are symmetric" invariant only in the sense that it drops both directions
symmetrically; callers that need perception again call `perceive`.

**Vina equivalent.** `model` in `model.h` is much larger (it also owns the
ligand tree, the pair lists, the grids and the forces); `Molecule` is the
*narrow* equivalent of its atom storage plus `model::assign_bonds` /
`model::assign_types` / `model::center`. OpenDocking splits the rest of Vina's
`model` into `MovableModel`, `Shared` and `System`.

---

### `Pair`

```rust
// crates/dock-core/src/molecule.rs
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
    pub fn new(a: usize, b: usize) -> Pair;
}
```

**Field by field.** Two indices, order-normalised. The ordering is not cosmetic:
`eval_pairs` writes `+grad` into `forces[a]` and `-grad` into `forces[b]`, so a
canonical order makes the accumulated gradient independent of how the pair list
was built, and lets the container be sorted/deduplicated.

**Invariants.** `a < b`, both indices are valid movable-atom indices, and the
pair is present at most once in a given list. Pair lists are only built between
atoms in *different* rigid frames and at least four bonds apart, so a rigid
fragment never interacts with itself.

**Vina equivalent.** `interacting_pair` in `model.h`
(`sz type_pair_index; sz a; sz b;`), produced by `model::initialize_pairs`.
Vina keeps the cached type-pair index in the same record; OpenDocking computes
the pair contribution from the two atoms instead, so `Pair` carries only the two
indices. Vina keeps two lists (the normal one and the macrocycle-closure one);
so does OpenDocking (`intra_pairs` and `glue_pairs` inside `Shared`).

---

## Kinematics

```text
The kinematic model, in one line:
    coords[i] = origin(frame) + orientation(frame) * local[i]
```

### `Frame`

```rust
// crates/dock-core/src/kinematics.rs
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
    pub fn local_to_lab(&self, v: DVec3) -> DVec3;            // origin + orientation * v
    /// Express a direction of the local frame in the laboratory frame.
    pub fn local_to_lab_direction(&self, v: DVec3) -> DVec3;  // orientation * v
}
```

**Field by field.** A rigid transform without scale. Points transform with the
origin translation; *directions* (in particular the torsion axis) must not, which
is why the two helpers exist.

**Invariant.** `orientation` is a unit quaternion (`length() == 1` within
floating-point error); the transform is an isometry, so all interatomic
distances inside a frame are preserved exactly (pinned by
`rigid_body_transform_is_an_isometry`).

**Vina equivalent.** `frame` in `tree.h` (`vec origin;` plus both a quaternion
(`qt orientation_q`) and its cached rotation matrix (`mat orientation_m`), with
`local_to_lab` and `local_to_lab_direction` helpers of exactly the same shape),
combined with the quaternion helpers in `quaternion.cpp` (`quaternion_to_r3`,
`angle_to_quaternion`). Vina materialises and caches the matrix per frame;
OpenDocking stores only the quaternion and multiplies it into the coordinates
directly. The rotation *axis* is a separate Vina type (`axis_frame`, and its
`segment` subclass), which `Node` folds into the same struct.

---

### `FrameKind`

```rust
// crates/dock-core/src/kinematics.rs
/// Which kind of degree of freedom a node carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// 6 DOF: translation + rotation of a whole rigid cluster.
    Rigid,
    /// 1 DOF: rotation about the node's axis.
    Torsion,
}
```

**Semantics.** A `Rigid` node owns a contiguous block of atoms that move
together; a `Torsion` node owns a block that rotates about its axis. A ligand
tree has exactly one `Rigid` node (the root) and one `Torsion` node per
rotatable bond; a flexible-residue tree has a pinned root whose kind is
`Torsion` with `absolute = true`.

**Invariant.** Only the root of a `TreeKind::Ligand` tree is `Rigid`; all other
nodes are `Torsion`. `assign_torsions` numbers the `Torsion` nodes in pre-order
and only those.

**Vina equivalent.** `rigid_body` vs `segment`/`first_segment` in `tree.h`;
Vina distinguishes them by type, OpenDocking by this tag (the children are
uniformly typed so the recursion is a single `Vec<Node>`).

---

### `Node`

```rust
// crates/dock-core/src/kinematics.rs
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
    pub fn rigid_root(atoms: (usize, usize)) -> Node;
}
```

**Field by field.**

* `origin` / `orientation` are the *current* laboratory-frame frame. For the
  rigid root, `origin` is the ligand's `LigandConf::position` and `orientation`
  is its quaternion; for a torsion segment they are recomputed on every
  forward-kinematics pass from the parent frame, the relative origin and the
  torsion angle.
* `axis` is the unit rotation axis in the laboratory frame, computed once at
  build time as `normalize(coords[attachment] - coords[parent])` and then
  re-expressed in the parent frame on each pass. If the two atoms coincide
  (degenerate file), the axis falls back to `+Z` (`EPSILON` guard in
  `BuildCtx::build`).
* `rel_origin` / `rel_axis` are the same quantities expressed in the *parent*
  frame; they are the persistent description of the segment, constant for the
  whole run. `orientation` and `axis` are the derived, current values.
* `absolute` marks the pinned root of a flexible-residue tree: its
  `orientation` is recomputed from its own `axis` and the first torsion instead
  of being inherited from a parent.
* `atoms` is a half-open range into the flat movable atom array. It contains the
  frame's own atoms *and* the attachment atoms of all its child branches —
  because an attachment atom lies on the child's rotation axis and therefore
  never moves with the child. This is exactly AutoDock Vina's mobility
  convention and it is what makes the nesting unambiguous.
* `torsion` is the index into `LigandConf::torsions` that drives this node,
  assigned in pre-order by `assign_torsions`. It is `None` for a rigid root.

**Invariants.**

* The atom ranges of a node and all its descendants are pairwise disjoint and
  their union is the tree's own `atoms` range.
* `axis` is unit length (or `+Z` for a degenerate branch).
* For `kind == Rigid`, `torsion == None` and `absolute == false`; for
  `TreeKind::Flex`'s root, `absolute == true`.
* The number of torsion-indexed nodes equals `KinematicTree::num_torsions`, and
  `apply_ligand` asserts (in debug builds) that the torsion cursor consumed
  exactly `num_torsions` values.

**Vina equivalent.** `frame` + `atom_range` + `atom_frame` + `rigid_body` +
`axis_frame` + `segment` + `first_segment` in `tree.h`. Vina's `segment`
additionally holds a `relative_axis`/`relative_origin` pair (OpenDocking's
`rel_origin`/`rel_axis`) and a pointer to the parent; OpenDocking relies on the
recursion instead of parent pointers, and keeps the child list by value.

---

### `TreeKind`

```rust
// crates/dock-core/src/kinematics.rs
/// Whether a `KinematicTree` describes a ligand (free rigid body) or a
/// flexible receptor side chain (pinned root).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TreeKind {
    /// A ligand: the root is a `FrameKind::Rigid` node.
    Ligand,
    /// A flexible residue: the root is an absolute torsion segment.
    Flex,
}
```

**Invariant.** `TreeKind::Ligand` implies a rigid root and a
`LigandConf`-driven application; `TreeKind::Flex` implies an absolute root and a
torsion-vector-driven application. `apply_ligand` and `apply_flex` both
`debug_assert` on the match.

**Vina equivalent.** `ligand` vs `residue` in `model.h`; Vina picks the
behaviour from the object type, OpenDocking from this tag.

---

### `KinematicTree`

```rust
// crates/dock-core/src/kinematics.rs
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
    pub fn num_atoms(&self) -> usize;

    /// Forward kinematics for a ligand: apply `conf` and refresh `coords`.
    pub fn apply_ligand(&mut self, conf: &LigandConf, coords: &mut [DVec3]);

    /// Forward kinematics for a flexible residue.
    pub fn apply_flex(&mut self, torsions: &[f64], coords: &mut [DVec3]);

    /// Accumulate this tree's DOF gradient.
    pub fn accumulate_gradient(
        &self,
        coords: &[DVec3],
        forces: &[DVec3],
        g: &mut [f64],
        rigid_slots: Option<(usize, usize)>,
        torsion_base: usize,
    );
}

/// Build a ligand kinematic tree from its parsed topology.
pub fn build_ligand_tree(top: &TopologyNode, coords: &[DVec3], n_movable: usize)
    -> (KinematicTree, usize);

/// Build a flexible-residue (or flexible-side-chain) kinematic tree.
pub fn build_flex_tree(top: &TopologyNode, coords: &[DVec3], n_movable: usize)
    -> KinematicTree;
```

**Field by field.**

* `root` owns the whole tree by value.
* `num_torsions` is the number of torsion degrees of freedom (for a flex tree it
  counts the pinned root segment as well, so it is `n_tors + 1`).
* `local` is the *reference geometry*: every atom's coordinates expressed in the
  frame that owns it, frozen at build time. It has one entry per movable atom
  index (including entries for atoms this particular tree does not own, which
  stay zero and are never read).
* `atoms` is the half-open range of movable indices this tree spans. For a
  ligand it is the union of all node ranges — which is *not* necessarily
  `(0, n_movable)` when the movable array also holds flexible residues; for a
  flex tree it is `(0, n_movable)`.

**The forward-kinematics recurrence** (per `Node::apply_segment`):

```text
t = torsions[cursor]; cursor += 1
origin      = parent.local_to_lab(rel_origin)
axis        = parent.local_to_lab_direction(rel_axis)
orientation = normalize(angle_to_quaternion(axis, t) * parent.orientation)
coords[i]   = origin + orientation * local[i]        for i in node.atoms
```

and for the root:

```text
origin = conf.position; orientation = conf.orientation
coords[i] = origin + orientation * local[i]          for i in root.atoms
```

**Invariants.**

* `local.len() == MovableModel::coords.len()` for the ligand tree, and the
  topology must reference every movable atom exactly once
  (`debug_assert_eq!(order.len(), n_movable)` in both builders).
* `apply_ligand` is exactly the identity map at `LigandConf::null(...)`
  (identity orientation, zero torsions), which is why the search can start from
  the parsed structure.
* Every application is an isometry on the atoms of one frame; the *relative*
  geometry of two atoms in different frames changes only through the torsions
  between them.
* Torsion angles are consumed in pre-order, mirroring Vina's `flv::iterator`
  traversal.

**Vina equivalent.** `tree<segment>` / `branch` and `heterotree<rigid_body>` /
`flexible_body` / `main_branch` in `tree.h`, plus the frame list vector
(`flv`) machinery. Vina's `conf::apply` performs the same forward pass;
OpenDocking keeps the reference coordinates in a flat `local` vector instead of
per-frame arrays.

---

### `TopologyNode`

```rust
// crates/dock-core/src/kinematics.rs
/// Input topology needed to build a `KinematicTree`.
///
/// Atom entries are *global movable atom indices*. The parent atom of a branch
/// may be immobile (a receptor atom), so its position is carried as a
/// coordinate rather than as an index.
#[derive(Debug, Clone, Default)]
pub struct TopologyNode {
    /// Atoms owned by this frame, in file order.
    pub atoms: Vec<usize>,
    /// Position within `atoms` of the atom that remains fixed relative to the
    /// parent frame (the `BRANCH a b` "b" atom). Only set by the raw PDBQT
    /// parser; a flattened topology records `attachment` instead.
    pub immobile: Option<usize>,
    /// Global movable index of this branch attachment atom. The atom itself
    /// belongs to the *parent* frame: it lies on the rotation axis and therefore
    /// never moves, which is exactly AutoDock Vina's mobility convention.
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
    /// `true` when this branch actually rotates atoms.
    pub fn has_segment(&self) -> bool;
    /// Vina's `essentially_empty`.
    pub fn is_empty_branch(&self) -> bool;
}
```

**Field by field.** `atoms` is the frame's own movable atoms; `children` are the
torsion branches leaving the frame. `attachment` is the `b` atom of
`BRANCH a b`: it is the *rotation centre* and belongs to the parent frame, which
is why the tree builder adds it to the parent's range and not to the child's.
`axis_root`/`axis_root_atom` describe the `a` atom, the far end of the rotation
axis; the axis is `normalize(attachment_coords - axis_root)`. The two `coord`
fields exist so that a flexible residue can rotate about an axis defined by
*immobile* receptor atoms that have no movable index at all.

**Invariants.**

* For every node in a *flattened* topology: `attachment.is_some()` and
  `axis_root.is_some()` whenever `has_segment()` is true.
* `has_segment()` is true iff the branch owns at least one atom beyond its
  attachment atom or has a descendant that does; empty branches are dropped at
  build time, and the PDBQT parser also decides from this predicate whether a
  `BRANCH` block creates a rotor at all.
* A flattened `TopologyNode` never has `immobile == Some(..)` (the flattener
  clears it and records `attachment` instead); only the raw parser's internal
  nodes carry it.

**Vina equivalent.** The `pdbqt_initializer` / `parse_pdbqt.cpp` topology
construction and Vina's branches; the `BRANCH a b` handling (including the
"essentially empty branch" rule) is the direct reference, from AutoDock Vina's
`parse_pdbqt.cpp` and `model.cpp`.

---

## Conformations and the movable model

### `LigandConf`

```rust
// crates/dock-core/src/kinematics.rs
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
    pub fn null(position: DVec3, num_torsions: usize) -> LigandConf;
}
```

**Field by field.** `position` is the position of the *root atom*, i.e. the
frame origin of the rigid body — not the molecular centroid and not the box
centre. `orientation` is the rigid-body rotation about that same point, which is
why the analytic torque is accumulated about `root.origin` (see
[`SCORING.md`](SCORING.md#rigid-body-projection)). `torsions` has exactly
`KinematicTree::num_torsions` entries, in pre-order.

**Invariants.** `orientation` is unit length and is kept normalised by
`quaternion_increment`; every torsion is wrapped into `(-pi, pi]` by
`normalize_angle` and by `Conf::increment_flat`; `torsions.len()` equals the
tree's `num_torsions` and the layout's `ligands[0]`. The kernel docks exactly one
ligand per system in this release, so `Conf::ligands` has length ≤ 1 and
`MovableModel::find_ligand` returns `Some(0)` or `None`.

**Vina equivalent.** `ligand_conf` in `conf.h`, which holds a `rigid_conf`
(`vec position; qt orientation;`) plus an `flv torsions` vector. Vina's
`ligand_change` composes the same two halves with `rigid_change`
(`vec position; vec orientation;` — a translation and a rotation vector), and
`rigid_conf::increment` calls `quaternion_increment` with the rotation vector
exactly as this struct does.

---

### `LigandChange`

```rust
// crates/dock-core/src/kinematics.rs
/// A ligand degree-of-freedom *change* (translation, rotation vector, torsions).
#[derive(Debug, Clone, PartialEq)]
pub struct LigandChange {
    /// Translation increment (Å).
    pub position: DVec3,
    /// Rotation-vector increment (radians).
    pub orientation: DVec3,
    /// Torsion increments (radians).
    pub torsions: Vec<f64>,
}

impl LigandChange {
    /// Zero change with room for `num_torsions` torsions.
    pub fn zeros(num_torsions: usize) -> LigandChange;
}
```

**Field by field.** A *delta*, not a state: `position` is an additive
translation in Å, `orientation` is a **rotation vector** (exponential-map
increment: direction = axis, length = angle in radians), and `torsions` are
additive angle increments. The rotation vector exists so that BFGS can work in a
flat vector space; the increment is turned into a quaternion by
`quaternion_increment`.

**Invariant.** `torsions.len()` matches the conformation it is applied to; the
flat buffer that BFGS manipulates holds exactly the six rigid-body components
followed by the torsions.

**Vina equivalent.** `ligand_change` in `conf.h` (`vec rigid; flv torsions;`)
combined with `rigid_change`; Vina's `rigid_change` stores the same
six-component translation + rotation-vector increment.

---

### `Conf`

```rust
// crates/dock-core/src/kinematics.rs
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
    pub fn null(positions: &[DVec3], layout: &DofLayout) -> Conf;

    /// Apply a flat degree-of-freedom delta (`delta`) scaled by `factor`.
    pub fn increment_flat(&mut self, layout: &DofLayout, delta: &[f64], factor: f64);

    /// Randomise every degree of freedom inside the search box.
    pub fn randomize(&mut self, corner1: DVec3, corner2: DVec3, rng: &mut Rng);

    /// Reset torsions and rotation to the identity, keeping positions.
    pub fn set_to_null(&mut self);

    /// True when two conformations share their torsions within `cutoff` and
    /// their rigid bodies within the position/orientation cut-offs.
    pub fn too_close(
        &self, other: &Conf,
        torsion_cutoff: f64, position_cutoff: f64, orientation_cutoff: f64,
    ) -> bool;
}
```

**Field by field.** One `LigandConf` per ligand plus one torsion vector per
flexible residue. The flat DOF vector that the optimiser sees is the
concatenation

```text
[ ligand 0: x y z  wx wy wz  t0 .. tn-1 ] [ ligand 1: ... ] [ flex 0: t0 .. tk-1 ] [ ... ]
```

and `increment_flat` walks that layout in exactly this order: it adds the
translation, applies the quaternion increment to the rotation, and adds each
torsion with `normalize_angle` (both the increment and the result are wrapped).
`randomize` draws a uniform point in the box, a uniform orientation and uniform
angles; `state_to_null` — `set_to_null` — keeps positions but resets rotation
and torsions.

**Invariants.**

* `ligands.len() == layout.ligands.len()` and `flex[k].len() ==
  layout.flex[k]`; `increment_flat` `debug_assert`s that it consumed exactly
  `layout.num_floats()` values.
* Torsions are always inside `(-pi, pi]`.
* `increment_flat` is written so that a finite difference along one flat
  direction is exactly the analytic derivative convention (the finite-difference
  test in `kinematics.rs` differentiates through `increment_flat`, not through a
  hand-written perturbation).

**Vina equivalent.** `conf` in `conf.h` (`boost::ptr_vector<ligand_conf>
ligands; ptr_vector<residue_conf> flex;` plus `operator+=` for changes and
`randomize()`); `Conf::too_close` corresponds to Vina's `conf::operator==` /
RMSD-based dedup predicates (`distance`, `distance_sqr`).

---

### `DofLayout`

```rust
// crates/dock-core/src/kinematics.rs
/// The degree-of-freedom layout of a docking system.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct DofLayout {
    /// Number of torsions per ligand.
    pub ligands: Vec<usize>,
    /// Number of flexible residues.
    pub flex: Vec<usize>,
}

impl DofLayout {
    /// Total number of floating-point degrees of freedom.
    pub fn num_floats(&self) -> usize;              // sum(6 + n) + sum(flex)

    /// Offset of the ligand torsion block.
    pub fn ligand_torsion_offset(&self, ligand: usize) -> usize;

    /// Offset of the first ligand's translation block.
    pub fn ligand_position_offset(&self, ligand: usize) -> usize;

    /// Total number of torsion degrees of freedom (excluding the rigid body).
    pub fn num_torsions_total(&self) -> usize;

    /// Offset of the flexible-residue torsion block.
    pub fn flex_torsion_offset(&self, flex: usize) -> usize;
}
```

**Field by field.** Two counts per entity kind. `num_floats` is the length of
every flat gradient and every flat delta buffer in the crate;
`ligand_position_offset(0)` and `ligand_torsion_offset(0)` are the translation
and rotation slots that `MovableModel::dof_gradient` writes;
`num_torsions_total` is what `DockResult::num_dof` reports.

**Invariants.** `num_floats() == 6 * ligands.len() + ligands.iter().sum() +
flex.iter().sum()`; every offset is in range; the layout is derived from the
trees once at build time and never changes afterwards. One unit test pins the
round trip (`change_flat_layout_round_trips`).

**Vina equivalent.** There is no dedicated struct: Vina's `conf_size`
(`conf.h`) computes the same sizes (`conf_size::operator()(const conf&)`) and
`conf::operator+=` uses the same ordering. OpenDocking makes the offsets
explicit because the gradient is a flat `Vec<f64>` shared by BFGS.

---

### `MovableModel`

```rust
// crates/dock-core/src/kinematics.rs
/// The movable part of a docking system: ligand plus flexible residues.
#[derive(Debug, Clone)]
pub struct MovableModel {
    /// Movable atoms (ligand first, then the flexible residues).
    pub atoms: Vec<Atom>,
    /// Current laboratory coordinates.
    pub coords: Vec<DVec3>,
    /// Cartesian energy gradient `dE/dx`, refreshed by `accumulate_gradient`.
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
    pub fn num_atoms(&self) -> usize;
    pub fn is_atom_in_ligand(&self, i: usize) -> bool;
    pub fn find_ligand(&self, i: usize) -> Option<usize>;

    /// Apply a conformation, i.e. run the whole forward kinematics.
    pub fn apply(&mut self, conf: &Conf);

    /// The identity conformation for the *current* coordinates.
    pub fn initial_conf(&self) -> Conf;

    /// Project the Cartesian gradient onto the degrees of freedom.
    pub fn dof_gradient(&self) -> Vec<f64>;

    /// Gyration radius of the ligand (used to scale rotational mutations).
    pub fn ligand_gyration_radius(&self) -> f64;

    /// Heavy-atom coordinates of the movable atoms (used for RMSD).
    pub fn heavy_atom_coords(&self) -> Vec<DVec3>;

    /// Clone of the ligand atoms with their current coordinates.
    pub fn ligand_atoms(&self) -> Vec<Atom>;
}
```

**Field by field.**

* `atoms` are the movable atoms with their *templates* (typing, names); the
  ligand's atoms occupy `ligand.atoms`, then the flexible residues follow.
* `coords` is the live Cartesian state, rewritten by `apply`.
* `minus_forces` is the Cartesian gradient `dE/dx_i` — despite the name, which
  follows Vina's `m.minus_forces` member. It is recomputed from scratch at the
  start of every `System::evaluate` and is the input to the DOF projection.
* `ligand` / `flex` are the trees; `ligand_pairs` / `flex_pairs` are the
  explicit interaction lists used for the intra-molecular terms.
* `layout` describes the DOF vector.

**Invariants.**

* `atoms.len() == coords.len() == minus_forces.len()`.
* `ligand.local.len() == coords.len()` and the frame ranges are disjoint and
  cover `[0, coords.len())` for the ligand plus whatever the flex trees own.
* `apply` puts the tree back exactly as it found it: it uses `dummy_tree()`
  placeholders with `std::mem::replace` so that `coords` can be borrowed mutably
  at the same time as the tree.
* `dof_gradient()` returns a vector of length `layout.num_floats()`.
* `initial_conf()` reproduces the current coordinates through
  `apply` — the round-trip property the dock construction relies on.

**Vina equivalent.** Vina's `model` in `model.h` holds `atoms`, `coords`,
`minus_forces`, `ligands` (a `flexible_body`), `flex` (residues) and the two pair
lists in one object. OpenDocking splits that: the *immutable* half (receptor,
force field, grids, topology templates, `num_tors`) lives in
`Shared`, and the *mutable* half lives here, which is what makes the
`Arc<Shared>` + cloned-model threading model possible.

---

## Grids

### `GridDim`

```rust
// crates/dock-core/src/scoring/grid.rs
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
    pub fn empty() -> GridDim;
    pub fn span(&self) -> f64;        // end - begin
    pub fn n_points(&self) -> usize;  // n_voxels + 1
}
```

**Field by field.** The half-open world interval `[begin, end]` and the voxel
count. The `+1` convention matters: `n_voxels` intervals need `n_voxels + 1`
sample points, and the indexing in `AffinityGrid::evaluate` (
`idx = ((x * ny + y) * nz + z) * nt + slot`) is over sample points.

**Invariants.** `span == spacing * n_voxels` for a grid-derived dimension;
`n_voxels == 0` disables the axis — used by `GridDim::empty()` and interpreted by
both the grid and the exact scorer as "no out-of-box penalty on this axis", which
is how `odock score` and the scoring unit tests disable the box entirely.

**Vina equivalent.** `grid_dim` in `grid_dim.h` (`fl begin; fl end; sz
n_voxels;`), including `span()` and `enabled()`; `grid_dims` is
`boost::array<grid_dim, 3>`.

---

### `GridBox`

```rust
// crates/dock-core/src/scoring/grid.rs
/// The axis-aligned search box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GridBox {
    /// Box centre (Å).
    pub center: DVec3,
    /// Requested edge lengths (Å).
    pub size: DVec3,
    /// Grid spacing (Å); 0.375 is the AutoDock convention.
    pub spacing: f64,
    /// Round voxel counts up to even numbers, as required by `.map` files.
    pub force_even_voxels: bool,
}

impl Default for GridBox {
    fn default() -> Self {
        GridBox { center: DVec3::ZERO, size: DVec3::splat(22.5), spacing: 0.375,
                  force_even_voxels: false }
    }
}

impl GridBox {
    pub fn new(center: DVec3, size: DVec3, spacing: f64) -> GridBox;
    pub fn with_even_voxels(mut self, even: bool) -> GridBox;

    /// Derive a box that contains a molecule of the given half-extents plus
    /// `buffer` Å of padding (Vina's `grid_dimensions_from_ligand`).
    pub fn around(center: DVec3, half_extents: DVec3, buffer: f64, spacing: f64) -> GridBox;

    pub fn dims(&self) -> [GridDim; 3];
    pub fn corner1(&self) -> DVec3;
    pub fn corner2(&self) -> DVec3;
    pub fn volume(&self) -> f64;
}
```

**Field by field.** `center` and `size` are the *requested* box; `dims()` turns
them into the *realised* box by, per axis,

```text
n       = max(1, ceil(size[i] / spacing))
n       = n + (n % 2)                     if force_even_voxels
span    = spacing * n
begin   = center[i] - span / 2
end     = begin + span
```

so the realised box is never smaller than requested, and always an exact
multiple of the spacing. `corner1`/`corner2` are the rounded corners, and they
are the two arguments `Conf::randomize` uses to draw starting positions.

**Invariants.** `spacing > 0` (the Python layer and the PyO3 constructor both
enforce it before the box reaches Rust); `size[i] > 0`; `dims()[i].span() >=
size[i] - 1e-9`; `volume() == product of spans`. `force_even_voxels` is required
for `.map` output and must be turned off for normal docking, where an odd voxel
count is fine.

**Vina equivalent.** The box geometry is Vina's `grid_dims` plus
`grid_dimensions_from_ligand` (`vina.cpp`) and the `--center_*` / `--size_*`
handling. The `spacing = 0.375` default is the AutoDock convention Vina
inherits.

---

### `AffinityGrid`

```rust
// crates/dock-core/src/scoring/grid.rs
/// A pre-computed receptor affinity grid.
#[derive(Clone)]
pub struct AffinityGrid {
    /// Per-axis geometry.
    pub dims: [GridDim; 3],
    /// Grid spacing actually used (Å).
    pub spacing: f64,
    /// Out-of-box penalty slope.
    pub slope: f64,
    /// Atom types stored, in slot order.
    pub types: Vec<XsType>,
    /// `slot[t]` is the storage slot of X-Score type `t`, or `usize::MAX`.
    slot: [usize; XsType::COUNT],        // private
    /// Flat `[x][y][z][slot]` energy storage.
    data: Vec<f64>,                      // private
}

impl AffinityGrid {
    pub fn new(box_: &GridBox, slope: f64) -> AffinityGrid;
    pub fn from_dims(dims: [GridDim; 3], spacing: f64, slope: f64) -> AffinityGrid;
    pub fn is_initialized(&self) -> bool;         // !types.is_empty()
    pub fn shape(&self) -> [usize; 3];            // sample points per axis
    pub fn num_points(&self) -> usize;
    pub fn memory_mb(&self) -> usize;
    pub fn has_types(&self, types: &[XsType]) -> bool;
    pub fn available_types(&self) -> &[XsType];

    /// Build the maps for `types` from the receptor atoms (rayon over x).
    pub fn populate(&mut self, receptor: &[Atom], sf: &dyn ScoringFunction,
                    types: &[XsType]) -> usize;

    /// Upload a raw `[x][y][z][slot]` f32 map built elsewhere (the GPU backend).
    pub fn upload(&mut self, values: &[f32]) -> Result<(), String>;

    pub fn eval(&self, atoms: &[Atom], coords: &[DVec3], v: f64) -> f64;
    pub fn eval_from(&self, atoms: &[Atom], coords: &[DVec3], from: usize, v: f64) -> f64;
    pub fn eval_deriv(&self, atoms: &[Atom], coords: &[DVec3], v: f64,
                      forces: &mut [DVec3]) -> f64;
    pub fn is_in_grid(&self, atoms: &[Atom], coords: &[DVec3], margin: f64) -> bool;
    pub fn write_maps(&self, prefix: &Path) -> std::io::Result<Vec<PathBuf>>;
}

/// Map an atom type onto the grid slot it uses.
pub fn grid_type(t: XsType) -> Option<XsType>;

/// The X-Score types that can be stored in a grid, in canonical order.
pub const GRID_TYPES: [XsType; 19];
```

**Field by field.**

* `dims`, `spacing` and `slope` describe the geometry and the out-of-box penalty
  (`DEFAULT_SLOPE = 1e6`).
* `types` is the list of types that actually have a map, in slot order; only the
  types the ligand presents are built (`build_system` collects them), which is
  typically 6–10 instead of all 19.
* `slot[t]` is the inverse map: the storage slot of type `t`, or `usize::MAX`
  when there is none. `grid_type` is the collapser that makes a *requested* type
  addressable: closure carbons collapse onto `CH`/`CP`, and closure dummies and
  `W` map to `None`.
* `data` is one flat `f64` buffer in `[x][y][z][slot]` order.

**Invariants.**

* `data.len() == shape()[0] * shape()[1] * shape()[2] * types.len()` whenever
  `is_initialized()`.
* `slot[t] < types.len()` for every `t` with a map, and `slot[t] == usize::MAX`
  otherwise; `types` is duplicate-free.
* `has_types` treats a type with no grid slot (`grid_type` returns `None`) as
  satisfied, because such an atom is skipped by the evaluator anyway.
* `eval`, `eval_deriv` and `is_in_grid` accept *all* movable atoms and skip
  those with no slot, so a caller never has to pre-filter.
* `write_maps` refuses to write when any axis has an odd number of voxels,
  because the `.map` format requires even counts.

**Vina equivalent.** `cache` in `cache.h` + `cache.cpp` is the closest
structure (`m_grids[t]` per atom type, `m_slope`, `is_in_grid`, `populate`,
`write`). Vina's per-type storage is one `grid` object per type; OpenDocking uses
one interleaved buffer with an explicit slot table. `grid_type` corresponds to
Vina's `atom_type::xs` collapsing in `cache::populate` (`XS_TYPE_SIZE` handling
and the closure-carbon folding).

---

## Exact scoring

### `CellList`

```rust
// crates/dock-core/src/scoring/noncache.rs
/// A uniform, axis-aligned spatial hash over a set of points.
#[derive(Debug, Clone)]
pub struct CellList {
    origin: DVec3,               // private: lower corner of the bucket grid
    inv_cell: f64,               // private: 1 / cell
    cell: f64,                   // private: cell side length (Å)
    nx: usize, ny: usize, nz: usize,   // private: bucket counts
    /// Flat `x + nx*(y + ny*z)` bucket index.
    buckets: Vec<Vec<usize>>,    // private
    positions: Vec<DVec3>,       // private
}

impl CellList {
    /// Build a cell list over `positions`.
    pub fn new(positions: &[DVec3], cell: f64) -> CellList;

    pub fn len(&self) -> usize;
    pub fn is_empty(&self) -> bool;

    /// Invoke `f` for every indexed point within `radius` of `p`.
    pub fn for_each_in_radius<F: FnMut(usize)>(&self, p: DVec3, radius: f64, f: F);

    pub fn cell_size(&self) -> f64;
}
```

**Field by field.** A uniform grid: `origin` and `inv_cell` map a point to a
bucket, `buckets` holds the point indices of each cell and `positions` keeps the
coordinates so callers can do the exact distance test. The requested cell size is
clamped to `[1.5, 12.0]` Å and the bucket counts to at most 64 per axis
(`MAX_CELLS_PER_AXIS`), so neither a single-atom cloud nor a huge protein
produces a pathological structure.

**Invariants.** `for_each_in_radius` is **conservative**: it visits every point
within `radius` and possibly some points a little further away (up to
`radius + cell * sqrt(3)`), so the caller *must* re-test the exact distance. Both
call sites do (`NonCache::eval` tests `r2 < cutoff_sqr`; `AffinityGrid::populate`
tests `r2 > cutoff_sqr` and returns early). `buckets[i]` contains exactly the
points whose bucket index is `i`; every point is in exactly one bucket.

**Vina equivalent.** `szv_grid` in `szv_grid.h` — Vina's "spatial zero-variance"
grid, used by `non_cache` and `cache::populate` for exactly the same
neighbour query.

---

### `NonCache`

```rust
// crates/dock-core/src/scoring/noncache.rs
/// Exact ligand-receptor scoring with analytic gradients.
#[derive(Debug, Clone)]
pub struct NonCache {
    /// Receptor atoms (immobile).
    pub receptor: Vec<Atom>,
    /// Spatial index over `receptor`.
    pub cells: CellList,
    /// The force-field cutoff in Å.
    pub cutoff: f64,
    /// The largest force-field cutoff in Å.
    pub max_cutoff: f64,
    /// Out-of-box penalty slope (Vina uses `1e6`).
    pub slope: f64,
    /// Search-box dimensions; `n_voxels == 0` disables the out-of-box penalty.
    pub dims: [GridDim; 3],
}

impl NonCache {
    pub fn new(receptor: Vec<Atom>, sf: &dyn ScoringFunction,
               dims: [GridDim; 3], slope: f64) -> NonCache;

    /// Energy of every movable atom against the receptor, optionally
    /// accumulating the Cartesian gradient in `forces`.
    pub fn eval(&self, sf: &dyn ScoringFunction, movable: &[Atom], coords: &[DVec3],
                v: f64, ligand_only: bool, n_ligand: usize,
                forces: Option<&mut [DVec3]>) -> f64;

    /// `true` when every heavy movable atom lies inside the search box.
    pub fn within(&self, movable: &[Atom], coords: &[DVec3], margin: f64) -> bool;
}

/// Pairwise energy over an explicit pair list, optionally with gradient.
pub fn eval_pairs(sf: &dyn ScoringFunction, atoms: &[Atom], coords: &[DVec3],
                  pairs: &[crate::molecule::Pair], v: f64, cutoff: f64,
                  forces: Option<&mut [DVec3]>) -> f64;
```

**Field by field.** `receptor` and `cells` are the immobile partner and its
index; `cutoff`/`max_cutoff` are copied from the force field so the hot loop does
not call through a trait object; `slope` and `dims` describe the out-of-box
penalty. `eval` clamps each movable atom into the box first
(`clamp_to_box` returns the clamped position, the linear penalty and its
gradient), accumulates the pair energies inside the cutoff, applies the soft cap
`curl(·, v)` to the *sum over receptor pairs of one atom* (Vina's semantics,
pinned by `exact_eval_matches_a_brute_force_sum`) and adds the out-of-box
penalty. `eval_pairs` does the same for an explicit pair list, writing `+grad`
to the first and `-grad` to the second atom of each pair.

**Invariants.**

* The receptor is never moved; only the movable side is clamped.
* Movable atoms typed `XsType::W` and closure dummies are skipped and get a zero
  gradient when the force field is X-Score typed.
* The gradient convention is `forces[i] == dE/dx_i` (see `SCORING.md`); a positive
  component means the energy grows when the atom moves in `+x`.
* The out-of-box penalty is exactly `slope * distance` per escaped axis, and its
  gradient is a constant `±slope` — pinned by
  `out_of_box_penalty_is_a_linear_ramp` (`forces[0].x == 1e6` for a probe 3 Å
  outside a box with slope `1e6`).

**Vina equivalent.** `non_cache` in `non_cache.cpp` (`eval`, `eval_deriv`,
`within`, the `out_of_bounds_penalty` accumulation, and the `curl` of each atom's
summed energy), plus `non_cache`'s pair-list work in `eval_intra` for
`eval_pairs`.

---

## The force field

### `Weights`

```rust
// crates/dock-core/src/scoring/mod.rs
/// Force-field term weights.
///
/// `terms` holds one weight per distance-dependent potential, in the order the
/// potentials are declared. `rot` is the *user-facing* torsional weight, i.e.
/// the value quoted in the papers: `E_final = E / (1 + rot * N_tors)` for
/// Vina/Vinardo and `E_final = E + rot * N_tors` for AD4.
#[derive(Debug, Clone, PartialEq)]
pub struct Weights {
    /// One weight per distance-dependent term.
    pub terms: Vec<f64>,
    /// Torsional weight.
    pub rot: f64,
}

impl Weights {
    pub fn vina_default() -> Weights;      // 6 terms, rot = 0.05846
    pub fn vinardo_default() -> Weights;   // 5 terms, rot = 0.05846
    pub fn ad4_default() -> Weights;       // 5 terms, rot = 0.2983
    pub fn default_for(choice: SfChoice) -> Weights;
    pub fn num_terms(&self) -> usize;
    pub fn set_term(&mut self, index: usize, value: f64) -> bool;
}
```

**The numbers** (from Vina's `vina.h` defaults, quoted in
[`SCORING.md`](SCORING.md#numeric-definitions)):

| Force field | `terms` | `rot` |
|---|---|---|
| Vina | `[-0.035579, -0.005156, 0.840245, -0.035069, -0.587439, 50.0]` | 0.05846 |
| Vinardo | `[-0.045, 0.8, -0.035, -0.600, 50.0]` | 0.05846 |
| AD4 | `[0.1662, 0.1209, 0.1406, 0.1322, 50.0]` | 0.2983 |

**Invariants.** `terms.len() == ScoringFunction::num_terms()` and the weight at
index `k` belongs to the term at index `k` of the same force field — the pairing
is positional and is established by the constructors, so a custom `Weights` must
keep the length. `set_term` returns `false` for an out-of-range index instead of
panicking.

**Vina equivalent.** `scoring_function::set_vina_weights` /
`set_vinardo_weights` / `set_ad4_weights` in `vina.h` (the argument names are
literally `weight_gauss1`, `weight_gauss2`, `weight_repulsion`,
`weight_hydrophobic`, `weight_hydrogen`, `weight_glue`, `weight_rot`), together
with the `conf_independent` evaluators in `conf_independent.cpp`
(`num_tors_div` for Vina/Vinardo, `ad4_tors_add` for AD4) that consume `rot`.

---

### `Term`

```rust
// crates/dock-core/src/scoring/mod.rs
/// Which radii table a `Term::Gauss` should use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Radii { Vina, Vinardo }

/// A single distance-dependent potential.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Term {
    /// `exp(-(r/width)^2)` evaluated at the reduced distance plus an offset.
    Gauss { offset: f64, width: f64, cutoff: f64 },
    /// `r^2` for `r < 0` (a soft-sphere repulsion).
    Repulsion { offset: f64, cutoff: f64 },
    /// Smooth hydrophobic contact term.
    Hydrophobic { good: f64, bad: f64, cutoff: f64 },
    /// Non-directional hydrogen bond.
    HBond { good: f64, bad: f64, cutoff: f64 },
    /// Linear attraction used to "glue" macrocycle closure pseudo-atoms.
    LinearAttraction { cutoff: f64 },
    /// AD4 Lennard-Jones 12-6 van der Waals term.
    Ad4Vdw { smoothing: f64, cap: f64, cutoff: f64 },
    /// AD4 12-10 hydrogen-bond term.
    Ad4Hb { smoothing: f64, cap: f64, cutoff: f64 },
    /// AD4 distance-dependent-dielectric Coulomb term.
    Ad4Elec { cap: f64, cutoff: f64 },
    /// AD4 desolvation term.
    Ad4Solv { sigma: f64, solvation_q: f64, charge_dependent: bool, cutoff: f64 },
}

impl Term {
    pub fn cutoff(&self) -> f64;
    pub fn eval_xs(&self, t1: XsType, t2: XsType, r: f64, radii: Radii) -> f64;
    pub fn eval_xs_deriv(&self, t1: XsType, t2: XsType, r: f64, radii: Radii) -> (f64, f64);
    pub fn eval_ad4(&self, a: &Atom, b: &Atom, r: f64) -> f64;
    pub fn eval_ad4_deriv(&self, a: &Atom, b: &Atom, r: f64) -> (f64, f64);
}
```

**Field by field.** Every variant is a *parameter set*, not a state: it carries
exactly the constants its formula needs (offsets, widths, ramp bounds, smoothing
widths, caps, cutoffs). `eval_xs_deriv` and `eval_ad4_deriv` return the value and
`dE/dr` together so the energy and its derivative can never be computed from
different branches. The X-Score terms consult `XsType` flags (hydrophobic,
donor/acceptor, glue) and the radii table; the AD4 terms consult the `AtomKind`
table and the atom charges.

**Invariants.**

* Each variant returns `(0.0, 0.0)` for `r >= cutoff`, and the AD4 vdW/H-bond
  terms are mutually exclusive by the sign of `hb_depth` (a pair is one or the
  other, never both).
* `eval_xs*` never uses an AD4 variant and `eval_ad4*` never uses an X-Score
  variant; the mismatched arms return `(0.0, 0.0)`.
* `Term::LinearAttraction` fires only for a "glued" pair and only inside its
  (deliberately longer) cutoff.
* `cutoff()` is total over the variants, so the grid and the neighbour search can
  always ask for a bound.

**Vina equivalent.** The `Potential` base class and its subclasses in
`potentials.h`: `vina_gaussian`, `vina_repulsion`, `vina_hydrophobic`,
`vina_hbond`, `vinardo_gaussian`, `vinardo_repulsion`, `vinardo_hydrophobic`,
`ad4_vdw`, `ad4_hbond`, `ad4_electrostatic`, `ad4_solvation`, `linearattraction`.
OpenDocking fuses the `eval`/`eval_deriv` pair into one call per term and keeps
the parameters in an enum instead of a class hierarchy.

---

### `ScoringFunction`

```rust
// crates/dock-core/src/scoring/mod.rs
/// Summed energy components of a pose, in kcal/mol.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ScoreComponents {
    /// The value the search minimises, including the torsional penalty.
    pub total: f64,
    /// Ligand-receptor energy.
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
    pub fn affinity(&self) -> f64;   // == self.total
}

/// The common interface implemented by every force field.
pub trait ScoringFunction: Send + Sync + fmt::Debug {
    fn choice(&self) -> SfChoice;
    fn weights(&self) -> &Weights;
    fn num_terms(&self) -> usize;
    fn cutoff(&self) -> f64;
    fn max_cutoff(&self) -> f64;

    fn eval_term(&self, k: usize, t1: XsType, t2: XsType, r: f64) -> f64;
    fn eval_term_deriv(&self, k: usize, t1: XsType, t2: XsType, r: f64) -> (f64, f64);
    fn eval_term_atoms(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> f64;
    fn eval_term_atoms_deriv(&self, k: usize, a: &Atom, b: &Atom, r: f64) -> (f64, f64);

    fn conf_independent(&self, e: f64, num_tors: f64) -> f64;

    fn is_grid_capable(&self) -> bool { true }
    fn is_xs_typed(&self) -> bool { true }

    fn pair_energy(&self, a: &Atom, b: &Atom, r: f64) -> f64;
    fn pair_energy_deriv(&self, a: &Atom, b: &Atom, r: f64) -> (f64, f64);
    fn pair_energy_xs(&self, t1: XsType, t2: XsType, r: f64) -> f64;
}

/// X-Score-based force field driver shared by Vina and Vinardo.
#[derive(Debug, Clone)]
pub struct XsScoringFunction {
    choice: SfChoice,        // private
    weights: Weights,        // private
    terms: Vec<Term>,        // private
    radii: Radii,            // private
    cutoff: f64,             // private
    max_cutoff: f64,         // private
}

/// The AutoDock 4.2 force field.
#[derive(Debug, Clone)]
pub struct Ad4ScoringFunction {
    weights: Weights,        // private
    terms: Vec<Term>,        // private
}

/// Which force field to use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SfChoice { Vina, Vinardo, Ad4 }

/// Construct a force field from a choice and optional weights.
pub fn make_scoring_function(choice: SfChoice, weights: Option<Weights>)
    -> Arc<dyn ScoringFunction>;
```

**Field by field.**

* `ScoreComponents` is the *bookkeeping* of the affinity formula: `total` is what
  the search minimises and what is reported; `conf_independent` is
  `total - (inter + intra - unbound)`, i.e. the effect of the torsional rule;
  `unbound` is the ligand's intramolecular energy in isolation. For linear
  force fields it is `total - base` and is usually positive; for AD4 it is
  `rot * N_tors - intra` (see `docking.rs::evaluate`).
* `ScoringFunction` is *term-oriented*: the grid builder asks for one term at a
  time (`eval_term`), the exact scorer asks for the summed pair energy
  (`pair_energy` / `pair_energy_deriv`), and the AD4 path uses the
  `*_atoms` variants because it needs charges.
* `is_grid_capable` is `false` only for AD4 (its electrostatics and desolvation
  need per-atom charges and must be evaluated exactly).
* `is_xs_typed` is `false` only for AD4; it is what makes the hydrogen skip and
  the `XsType::W` sentinel meaningful.
* The two concrete implementations differ in their term vector (6, 5 and 5
  terms), their radius table and their `conf_independent` rule; nothing else.

**Invariants.**

* `weights().terms.len() == num_terms()`, and `cutoff() <= max_cutoff()`.
* `pair_energy` and `pair_energy_deriv` agree exactly with the term-by-term sum
  (the tests differentiate `pair_energy` and compare against
  `pair_energy_deriv`).
* `conf_independent` is linear in its first argument for every force field
  (division by a topology-only constant, or addition of a constant), so it cannot
  move the position of the minimum; that is why the search objective omits it.
* `ScoringFunction` is `Send + Sync`, which is what allows an `Arc<dyn
  ScoringFunction>` to be shared by every rayon task.

**Vina equivalent.** `ScoringFunction` in `scoring_function.h` (`eval(atom&,
atom&, fl)`, `eval(sz, sz, fl)`, `conf_independent`, `get_cutoff`,
`get_max_cutoff`), and its concrete Vina/Vinardo/AD4 instances built in
`vina.cpp`. The five `ScoreComponents` fields correspond exactly to the
`total`, `inter`, `intra`, `conf_independent` and `unbound` members of Vina's
`output_type` (`conf.h`), which `Vina::score` fills and the result table prints;
`ScoringFunction` itself only ever sees one number at a time.

---

## Search

### `Caps`

```rust
// crates/dock-core/src/search/mod.rs
/// Energy caps applied during the search (Vina's `hunt_cap`) and during final
/// scoring (Vina's `authentic_v`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Caps {
    /// Cap for the intra-ligand pair terms.
    pub intra: f64,
    /// Cap for the grid (ligand-receptor) term.
    pub grid: f64,
    /// Cap for the explicit intermolecular / flex pair terms.
    pub pairs: f64,
}

impl Caps {
    /// The loose caps used for reported energies (Vina's `authentic_v`).
    pub fn authentic() -> Caps;    // { intra: 1000.0, grid: 1000.0, pairs: 1000.0 }
    /// The tight caps used during the search (Vina's `hunt_cap`).
    pub fn hunt() -> Caps;         // { intra: 10.0, grid: 1.5, pairs: 10.0 }
    pub fn with_grid_cap(mut self, cap: f64) -> Caps;
}
```

**Field by field.** These are the `v` arguments of `math::curl`: a soft cap
`e -> v*e/(v+e)` applied to positive energies (and with a consistent `(v/(v+e))²`
factor to the gradient). During the search, tight caps (`hunt`) suppress
clashes so BFGS is not thrown off by a huge gradient; for the reported energies,
loose caps (`authentic`) make the number as close to the physical interaction
energy as possible without an overflow.

**Invariants.** `assert!(Caps::hunt().grid < Caps::authentic().grid)`
semantically: the hunt caps must be the tighter ones. Every evaluation passes the
cap set explicitly, so energy and gradient are always computed with the same
caps.

**Vina equivalent.** `hunt_cap` = `vec(10, 1.5, 10)` and `authentic_v = 1000`
in `vina.cpp` / `non_cache.cpp`; the `curl` helper is `curl.h`.

---

### `Pose`

```rust
// crates/dock-core/src/search/mod.rs
/// A docked pose.
#[derive(Debug, Clone, PartialEq)]
pub struct Pose {
    /// The conformation.
    pub conf: Conf,
    /// The energy the search minimised (the reported affinity).
    pub energy: f64,
    /// RMSD (lower bound) to the best pose.
    pub lower_bound: f64,
    /// RMSD (upper bound, atom-to-atom) to the best pose.
    pub upper_bound: f64,
    /// Full energy decomposition.
    pub components: ScoreComponents,
    /// Heavy-atom coordinates of the ligand.
    pub coords: Vec<DVec3>,
    /// Whether the ligand lies inside the search box.
    pub in_box: bool,
}

impl Pose {
    /// An empty, maximally bad pose.
    pub fn empty() -> Pose;   // energy = MAX_F, in_box = false
}
```

**Field by field.** A pose carries both the *state* (`conf`, `coords`) and the
*report* (`energy`, `components`, RMSDs, `in_box`). `coords` is the ligand's
heavy-atom coordinates in the kernel's internal order, which is the
representation the RMSD container needs. `lower_bound`/`upper_bound` are relative
to the *best* pose (0 for the best one) and are filled in only after the final
sort.

**Invariants.** `energy == components.total`; `coords` is the heavy-atom subset
of the ligand produced by the same conformation that produced `energy`;
`in_box` is evaluated with a zero margin. `Pose::empty()` is a sentinel with
`energy = MAX_F` that is always replaced before use.

**Vina equivalent.** `output_type` in `conf.h` is a close one-to-one match:
`struct output_type { conf c; fl e; fl lb; fl ub; fl intra; fl inter;
fl conf_independent; fl unbound; fl total; vecv coords; }` — the same state,
energy, RMSD pair, component breakdown and heavy-atom coordinates. The container
semantics (`add_to_output_container`, the RMSD dedup, the best-first sort) come
from `coords.cpp` / `monte_carlo.cpp`.

---

### `SearchParams`

```rust
// crates/dock-core/src/search/mod.rs
/// Tunable search parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchParams {
    /// Number of independent Monte-Carlo runs (Vina's `exhaustiveness`).
    pub exhaustiveness: usize,
    /// Number of poses to retain.
    pub num_poses: usize,
    /// RMSD below which two poses are considered identical (Å).
    pub min_rmsd: f64,
    /// Optional hard cap on the number of energy evaluations per run.
    pub max_evals: u64,
    /// Metropolis temperature (kcal/mol); Vina uses 1.2.
    pub temperature: f64,
    /// Mutation amplitude in Å (Vina uses 2.0).
    pub mutation_amplitude: f64,
    /// Maximum number of MC steps per run; `None` selects Vina's heuristic.
    pub global_steps: Option<usize>,
    /// Number of local-search steps; `None` selects `(25 + n_atoms)/3`.
    pub local_steps: Option<usize>,
    /// Random seed; `0` means "draw one from the OS entropy pool".
    pub seed: u64,
    /// Use the island-model genetic algorithm instead of parallel MC.
    pub use_island_ga: bool,
    pub islands: usize,        // islands for the GA
    pub population: usize,     // island population size
    pub generations: usize,    // generations per island
    pub elites: usize,         // elites carried over per generation
}

impl Default for SearchParams {
    fn default() -> Self {
        SearchParams {
            exhaustiveness: 8, num_poses: 9, min_rmsd: 1.0, max_evals: 0,
            temperature: 1.2, mutation_amplitude: 2.0,
            global_steps: None, local_steps: None, seed: 0,
            use_island_ga: false, islands: 4, population: 32,
            generations: 20, elites: 2,
        }
    }
}

impl SearchParams {
    /// Vina's heuristic: `70 * 3 * (50 + n_atoms + 10*n_dof) / 2`.
    pub fn global_steps_for(&self, num_movable_atoms: usize, num_dof: usize) -> usize;
    /// Vina's heuristic: `(25 + n_atoms)/3`.
    pub fn local_steps_for(&self, num_movable_atoms: usize) -> usize;
    /// Resolve a zero seed into an entropy-derived one.
    pub fn effective_seed(&self) -> u64;
}
```

**Field by field.** The MC/Vina half is `exhaustiveness`, `num_poses`,
`min_rmsd`, `temperature`, `mutation_amplitude`, `global_steps`, `local_steps`
and `max_evals`; the GA half is `use_island_ga`, `islands`, `population`,
`generations` and `elites`. `max_evals == 0` disables the evaluation budget.
`seed == 0` is the "draw one from the OS" sentinel.

**Invariants.** `exhaustiveness` is clamped to at least 1 at the point of use
(`params.exhaustiveness.max(1)`), `population` to at least 4, `islands` to at
least 1, `generations` to at least 1, and `elites` to `[1, population]`. Of the
defaults, `temperature = 1.2 kcal/mol` (600 K with `R = 2 cal/(K*mol)`),
`mutation_amplitude = 2.0`, the hunt caps `(10, 1.5, 10)` and both step-count
heuristics are Vina's own constants; `exhaustiveness = 8` and
`min_rmsd = 1.0` match the defaults of `Vina::global_search`
(`exhaustiveness=8, n_poses=20, min_rmsd=1.0, max_evals=0`), and `num_poses = 9`
is OpenDocking's default number of reported modes.

**Vina equivalent.** `monte_carlo` in `monte_carlo.h` (`max_evals`,
`global_steps`, `temperature`, `hunt_cap`, `min_rmsd`, `num_saved_mins`,
`mutation_amplitude`, `local_steps`), `parallel_mc` in `parallel_mc.h`
(`num_tasks`, `num_threads`, plus the nested `mc`) and the arguments of
`Vina::global_search(exhaustiveness, n_poses, min_rmsd, max_evals)` in `vina.h`.
The GA fields have no Vina counterpart; they follow AutoDock 4's Lamarckian
genetic algorithm (`ga_num_evals`, `ga_pop_size`, `ga_num_generations`,
`ga_elitism`).

---

### `EnergyModel`

```rust
// crates/dock-core/src/search/mod.rs
/// Everything the search needs from a docking system.
pub trait EnergyModel: Send {
    fn layout(&self) -> &DofLayout;
    fn num_movable_atoms(&self) -> usize;
    fn gyration_radius(&self) -> f64;
    fn eval_deriv(&mut self, conf: &Conf, caps: Caps, grad: &mut [f64]) -> f64;
    fn eval(&mut self, conf: &Conf, caps: Caps) -> f64;
    fn score(&mut self, conf: &Conf, caps: Caps) -> ScoreComponents;
    fn ligand_coords(&self) -> Vec<DVec3>;
    fn ligand_atoms(&self) -> Vec<Atom>;
    fn within_box(&self, margin: f64) -> bool;
}
```

**Field by field.** The trait is the *seam* between the search and the docking
system: the search only ever sees a DOF layout, an energy/gradient oracle and the
ligand's coordinates. `eval_deriv` returns the objective (`inter + intra -
unbound` for Vina/Vinardo, `inter + conf_independent` for AD4) and fills `grad`
with the flat DOF gradient.

**Invariants.** `grad.len() == layout().num_floats()`; `eval` and
`eval_deriv` return the same value for the same `(conf, caps)`;
`ligand_coords()` reflects the most recent evaluation. The trait exists so the
search can be unit-tested against cheap analytic surfaces — the `Toy` model in
the test module implements exactly the same contract.

**Vina equivalent.** There is no equivalent abstraction: Vina's `monte_carlo`
and `bfgs` templates are instantiated on `model`, which owns everything.
OpenDocking introduces the trait so `System` and test doubles are
interchangeable.

---

## Orchestration

### `Shared`

```rust
// crates/dock-core/src/docking.rs
/// Immutable scoring context shared by every search thread.
#[derive(Debug, Clone)]
pub struct Shared {
    /// The rigid receptor atoms.
    pub receptor: Vec<Atom>,
    /// Exact scorer over the receptor.
    pub noncache: NonCache,
    /// Optional affinity grid.
    pub grid: Option<AffinityGrid>,
    /// The force field.
    pub sf: Arc<dyn ScoringFunction>,
    /// Ligand atom templates (typing, names and *initial* coordinates).
    pub ligand_atoms: Vec<Atom>,
    /// Ligand atom line templates for PDBQT output.
    pub ligand_records: Vec<RecordAtom>,
    /// Ligand topology, for PDBQT output.
    pub top: TopologyNode,
    /// Rotatable bonds, for the torsional penalty.
    pub rotors: Vec<(Option<usize>, usize)>,
    /// Vina's `num_tors` value.
    pub num_tors: f64,
    /// Interaction pairs inside the ligand.
    pub intra_pairs: Vec<Pair>,
    /// Macrocycle closure pairs (long-range linear attraction).
    pub glue_pairs: Vec<Pair>,
    /// Search box, as requested.
    pub box_: GridBox,
    /// Search box, as rounded by the grid dimensions.
    pub dims: [GridDim; 3],
    /// Out-of-box penalty slope.
    pub slope: f64,
    /// Use the grid during the search.
    pub use_grid: bool,
}

impl Shared {
    /// Number of torsion segments in the ligand topology.
    pub fn num_torsions(&self) -> usize;
}
```

**Field by field.** Everything that does not change during a run: the receptor
and its exact scorer, the optional grid, the force field, the ligand's typing and
output templates, the topology, the rotor list, the torsion count, the two pair
lists, and the box geometry. `rotors` is `(parent movable index or None,
attachment movable index)` per rotatable bond; `None` means the parent is an
immobile receptor atom (flexible-residue case).

**Invariants.**

* Immutability: after `build_system` hands out `Arc<Shared>`, nothing mutates it
  while a search runs. `Docking::rebuild_grid` uses `Arc::make_mut`, which
  clones only if another reference exists.
* `grid.is_some()` implies `use_grid && sf.is_grid_capable()`; AD4 therefore
  always has `grid == None`.
* `num_tors == num_tors(ligand_atoms, rotors)` and is a multiple of 0.5.
* `intra_pairs` and `glue_pairs` are disjoint and reference only movable
  indices; `dims == box_.dims()`.

**Vina equivalent.** Vina's `model` (receptor atoms, the typed ligand, the pair
lists) fused with the `cache`/`non_cache` and `scoring_function` that `Vina`
owns, minus everything mutable. The split is what makes the `Arc<Shared>` +
cloned-model threading model possible.

---

### `System`

```rust
// crates/dock-core/src/docking.rs
/// A docking system: shared scoring context plus per-thread mutable state.
#[derive(Clone)]
pub struct System {
    /// Shared, immutable context.
    pub shared: Arc<Shared>,
    /// Movable atoms, coordinates and kinematic trees.
    pub movable: MovableModel,
    /// Force the exact (non-grid) interaction energy.
    pub force_exact: bool,
}

impl System {
    /// Build an independent clone for a search thread.
    pub fn fork(template: &System) -> System;

    pub fn is_inside_box(&self, margin: f64) -> bool;
    pub fn current_components(&mut self, conf: &Conf) -> ScoreComponents;
    pub fn current_ligand_atoms(&self) -> Vec<Atom>;
    pub fn current_ligand_coords(&self) -> Vec<DVec3>;
}

impl EnergyModel for System { /* forwards to `evaluate` and `objective` */ }

/// Build a `System` from receptor and ligand PDBQT text.
pub fn build_system(receptor_text: &str, ligand_text: &str, options: &DockOptions)
    -> Result<System>;

/// Enumerate the intra-ligand interaction pairs.
pub fn ligand_pairs(atoms: &[Atom], frames: &[usize], ligand_range: (usize, usize),
                    xs_typed: bool) -> (Vec<Pair>, Vec<Pair>);

/// Re-minimise a pose with the exact (non-grid) scorer.
pub fn refine_pose(system: &mut System, pose: &mut Pose, steps: usize) -> f64;
```

**Field by field.** `shared` is the ref-counted immutable half; `movable` is the
per-thread mutable half; `force_exact` switches the ligand-receptor term from the
grid to the exact scorer — the search runs with `false`, refinement and final
scoring with `true`.

**Invariants.**

* `System::fork` clones the `Arc` and the `MovableModel` and always resets
  `force_exact` to `false`, so a forked system starts from the template's
  geometry and cannot inherit a stale exact-mode flag.
* `force_exact == true` implies the reported energies come from the exact scorer
  (`DockResult::exact` mirrors `DockOptions::refine`).
* `movable` and `shared` are consistent: `movable.layout` matches the tree,
  `movable.ligand_pairs` is a copy of `shared.intra_pairs`, and every movable atom
  index used by the shared pair lists is in range.

**Vina equivalent.** `Vina` in `vina.h`, specifically the `(model, igrid*,
scoring_function, conf)` quartet that Vina passes around. OpenDocking packages it
as `Arc<Shared>` + `MovableModel` so it can be cloned cheaply per rayon task.

---

### `Docking` (and `DockOptions`)

```rust
// crates/dock-core/src/docking.rs
/// Out-of-box penalty slope (Vina uses `1e6`).
pub const DEFAULT_SLOPE: f64 = 1e6;

/// Options controlling a docking run.
#[derive(Clone)]
pub struct DockOptions {
    pub sf_choice: SfChoice,          // force field
    pub weights: Option<Weights>,     // weight override
    pub box_: GridBox,                // the search box
    pub use_grid: bool,               // grid during the search
    pub refine: bool,                 // re-minimise with the exact scorer
    pub search: SearchParams,         // search parameters
    pub slope: f64,                   // out-of-box penalty slope
    pub energy_range: f64,            // reporting window (kcal/mol)
}

/// A docking job.
#[derive(Debug)]
pub struct Docking {
    /// The system being docked.
    pub system: System,
    /// The options in force.
    pub options: DockOptions,
    /// The ligand's original PDBQT text.
    pub ligand_source: String,
    /// The receptor's original PDBQT text.
    pub receptor_source: String,
    /// The most recent result.
    pub result: Option<DockResult>,
}

impl Docking {
    pub fn from_strings(receptor_pdbqt: &str, ligand_pdbqt: &str, options: DockOptions)
        -> Result<Docking>;
    pub fn center_box_on_ligand(&mut self, buffer: f64);
    pub fn score(&mut self) -> ScoreComponents;
    pub fn run(&mut self) -> Result<DockResult>;
    pub fn rebuild_grid(&mut self) -> Result<()>;
    pub fn poses_pdbqt(&self, energy_range: f64) -> String;
}
```

**Field by field.** `DockOptions::use_grid` is *and*-ed with
`sf.is_grid_capable()` before it is stored, so a user cannot accidentally ask for
a grid with AD4. `energy_range` is the reporting window used when rendering
poses. `Docking` keeps the original PDBQT text so a caller can inspect what was
docked, and caches the last `DockResult`.

**Invariants.** `result` is `None` until `run` (or `score`) is called;
`run` rebuilds the grid first, so changing `options.box_` after construction is
safe and the search and the final scores always agree. `run` errors instead of
returning an empty pose list (`DockError::Invalid("the search produced no poses;
check the grid box and the ligand")`).

**Vina equivalent.** `Vina` in `vina.h` (`set_receptor`, `set_ligand`,
`set_ligand_from_file`, `compute_vina_maps`, `score`, `dock`, `poses`,
`write_poses`). OpenDocking's `DockOptions` bundles what Vina passes to
`Vina::global_search` (`exhaustiveness`, `n_poses`, `min_rmsd`, `max_evals`)
together with the `model`-level scoring settings (force field, weights, box,
grid choice) and the refinement switch, and `run` corresponds to `Vina::dock`.

---

### `DockResult`

```rust
// crates/dock-core/src/docking.rs
/// The result of a docking run.
#[derive(Debug, Clone)]
pub struct DockResult {
    /// The reported poses, best first.
    pub poses: Vec<Pose>,
    /// The random seed actually used (never `0`).
    pub seed: u64,
    /// Approximate affinity-grid memory in megabytes (0 when unused).
    pub grid_mb: usize,
    /// Number of grid points (0 when unused).
    pub grid_points: usize,
    /// The `num_tors` value used by the torsional penalty.
    pub num_tors: f64,
    /// Number of movable atoms.
    pub num_movable_atoms: usize,
    /// Number of torsion degrees of freedom (excluding the rigid body).
    pub num_dof: usize,
    /// Whether the reported energies come from the exact pairwise scorer.
    pub exact: bool,
}

impl DockResult {
    /// The best pose, if any.
    pub fn best(&self) -> Option<&Pose>;
    /// Poses within `range` kcal/mol of the best one.
    pub fn within_energy_range(&self, range: f64) -> Vec<&Pose>;
}
```

**Field by field.** `poses` is sorted by ascending energy and truncated to
`num_poses`; `seed` is the *resolved* seed (an entropy seed when the caller
passed 0) and is what makes a run reproducible; `grid_mb`/`grid_points` describe
the grid actually used; `num_tors` is the value that entered the torsional
penalty; `num_dof` is the number of torsion degrees of freedom (through
`DofLayout::num_torsions_total`); `exact` records whether the reported energies
were refined on the exact surface.

**Invariants.** `poses[0]` has the lowest energy; `poses[i].lower_bound` and
`upper_bound` are relative to `poses[0]`; `exact == options.refine`;
`grid_points == 0` iff no grid was built.

**Vina equivalent.** The populated `output_container`
(`boost::ptr_vector<output_type>`, `conf.h`) — the ranked list of results — plus
the `mode | affinity | dist from best mode` block that Vina writes into the
output PDBQT and the run metadata the `Vina` object holds after `dock()`.

---

## Worked example: the kinematic tree of a small molecule

The example below is the `chain_topology()` fixture from the unit tests in
`crates/dock-core/src/kinematics.rs`: four atoms `A`, `B`, `C`, `D` in a chain,
with one rotatable bond `B–C`. PDBQT would express it as `BRANCH 2 3`
(parent = atom 2 = `B`, attachment = atom 3 = `C`).

```text
Geometry (Å)                              Topology

        A          B          C          ROOT
   (-1,0,0)----(0,0,0)----(1,0,0)         ATOM 1  A   (-1, 0, 0)
                            |             ATOM 2  B   ( 0, 0, 0)
                            |            ENDROOT
                            D            BRANCH    2   3
                       (1,1,0)            ATOM 3  C   ( 1, 0, 0)
                                          ATOM 4  D   ( 1, 1, 0)
                                         ENDBRANCH   2   3
                                         TORSDOF 1

            rotation axis = normalize(C - B) = +x, through C
```

```text
The tree                                       Frames and atom ranges

  Node { kind: Rigid,                          root frame  (0..3)
         origin:      A = (-1, 0, 0) <-- LigandConf::position
         orientation: identity,
         axis:        +z (unused),
         atoms:       (0, 3),                  atoms 0=A, 1=B, 2=C
         torsion:     None,
         children:    [ child ] }
            |
            +-- Node { kind: Torsion,          torsion segment (3..4)
                       origin:      C = (1, 0, 0)
                       orientation: identity (recomputed each pass)
                       axis:        +x
                       rel_origin:  C - A = (2, 0, 0)
                       rel_axis:    (1, 0, 0)
                       absolute:    false
                       atoms:       (3, 4),     atom 3 = D
                       torsion:     Some(0),    <-- LigandConf::torsions[0]
                       children:    [] }
```

```text
Forward kinematics for conf = { position: A, orientation: identity,
                                torsions: [pi/2] }

  root frame:   coords[0] = A + I * (A - A) = (-1, 0, 0)   unchanged
                coords[1] = A + I * (B - A) = ( 0, 0, 0)   unchanged
                coords[2] = A + I * (C - A) = ( 1, 0, 0)   unchanged
                  -> A and B belong to the rigid root cluster and C is the
                     attachment atom the parent frame owns, so none of the
                     three moves when only the torsion changes

  torsion 0:    origin      = root.local_to_lab(rel_origin) = (1, 0, 0)
                axis        = root.local_to_lab_direction(rel_axis) = (1, 0, 0)
                orientation = normalize(q(+x, pi/2) * identity) = q(+x, pi/2)
                coords[3]   = origin + orientation * (D - C)
                            = (1, 0, 0) + q(+x, 90 deg) * (0, 1, 0)
                            = (1, 0, 1)          <-- (0,1,0) rotates to (0,0,1)

  Bond lengths are preserved: |D - C| = 1.000 before and after.
```

```text
Degrees of freedom (DofLayout { ligands: [1], flex: [] })

  flat index   meaning
  ----------   ---------------------------------------------
     0, 1, 2   translation  = d(position)
     3, 4, 5   rotation vector = d(omega), applied as
               quaternion_increment(orientation, d(omega))
        6      torsion 0 = d(torsions[0])

  num_floats()          = 6 + 1 = 7
  ligand_position_offset(0) = 0
  ligand_torsion_offset(0)  = 6
  num_torsions_total()      = 1
```

A second, nested example — the PDBQT fixture in `io/pdbqt.rs` — shows how a
branch inside a branch nests. Its flattened movable order is
`[C1, C2, O1, H1]`, the root frame owns `{C1, C2, O1}` (the attachment atom `O1`
belongs to the parent frame even though it is written inside the branch) and the
single torsion segment owns `{H1}`:

```text
ROOT                    index 0  C1   ┐
  ATOM 1 C1             index 1  C2   ├── root frame (Rigid), atoms (0, 3)
  ATOM 2 C2             index 2  O1   ┘      O1 = attachment of BRANCH 2 3
ENDROOT
BRANCH 2 3              index 3  H1   ─────  torsion 0, axis = O1 - C2
  ATOM 3 O1                                 segment owns only H1
  ATOM 4 H1
ENDBRANCH 2 3
TORSDOF 1

flattened order = [C1, C2, O1, H1]
rotors          = [(Some(1), 2)]        <-- (parent C2, attachment O1)
num_torsions    = 1
```

The order is exactly what `dock_core::io::pdbqt::Flattener::insert` produces and
exactly what `odock.pdbqt.flatten_tree` reproduces on the Python side, which is
why `Docking.pose_coords(k)[i]` is the position of `ligand_atom_names()[i]`.

---

## Where each structure lives in AutoDock Vina

| OpenDocking | AutoDock Vina | File |
|---|---|---|
| `Element` | `EL_TYPE_*` | `atom_constants.h` |
| `AdType`, `AtomKind`, `ATOM_KIND_DATA`, `ad_type_property` | `atom_type` (AD4 tagging), `atom_kind_data[]`, `string_to_ad_type`, `max_covalent_radius` | `atom_type.h`, `atom_constants.h` |
| `XsType`, `XS_VDW_RADII`, `XS_VINARDO_VDW_RADII`, `optimal_distance` | `atom_type` (XS tagging), `xs_vdw_radii[]`, `optimal_distance*`, `xs_is_hydrophobic/acceptor/donor`, `is_glued` | `atom_type.h`, `atom_constants.h` |
| `XsType::from_element` | `model::assign_types` | `model.cpp` |
| `Atom` | `atom_type` + `atom_base` + `atom` | `atom_type.h`, `atom_base.h`, `atom.h` |
| `Bond` | `bond` | `atom.h` |
| `Molecule` (atom storage + perception) | `model` (atom storage + `assign_bonds`) | `model.h`, `model.cpp` |
| `Pair` | `interacting_pair`, `model::initialize_pairs` | `model.h`, `model.cpp` |
| `Frame`, `Node`, `FrameKind` | `frame`, `atom_range`, `atom_frame`, `rigid_body`, `axis_frame`, `segment`, `first_segment` | `tree.h` |
| `KinematicTree`, `TreeKind`, `build_ligand_tree`, `build_flex_tree` | `tree`, `branch`, `heterotree`, `flexible_body`, `main_branch`, `flv` | `tree.h` |
| `TopologyNode` | the `pdbqt_initializer` branch structure | `parse_pdbqt.cpp`, `model.cpp` |
| `LigandConf`, `LigandChange`, `Conf`, `DofLayout` | `rigid_conf`, `rigid_change`, `ligand_conf`, `ligand_change`, `conf`, `conf_size` | `conf.h` |
| `MovableModel` | the mutable half of `model` (`atoms`, `coords`, `minus_forces`, `ligands`, `flex`) | `model.h` |
| `GridDim`, `GridBox` | `grid_dim`, `grid_dims`, `grid_dimensions_from_ligand` | `grid_dim.h`, `vina.cpp` |
| `AffinityGrid` | `cache` (per-type `grid` objects, `populate`, `write`, `is_in_grid`) | `cache.h`, `cache.cpp`, `grid.h` |
| `CellList` | `szv_grid` | `szv_grid.h`, `szv_grid.cpp` |
| `NonCache`, `eval_pairs` | `non_cache` (`eval`, `eval_deriv`, `eval_intra`, `within`) | `non_cache.cpp` |
| `Term`, `Weights`, `ScoringFunction`, `ScoreComponents` | `Potential` and its subclasses, `ScoringFunction`, the `set_*_weights` defaults | `potentials.h`, `scoring_function.h`, `vina.h` |
| `Caps`, `curl` | `hunt_cap`, `authentic_v`, `curl` | `vina.cpp`, `non_cache.cpp`, `curl.h` |
| `bfgs`, `SymMatrix` | `bfgs`, `triangular_matrix_index` | `bfgs.h`, `triangular_matrix_index.h` |
| `Pose`, `add_to_output_container` | `output_type`, `output_container`, `add_to_output_container` | `conf.h`, `monte_carlo.cpp` |
| `SearchParams`, `run_monte_carlo`, `parallel_monte_carlo` | `monte_carlo`, `parallel_mc`, `Vina::global_search` arguments | `monte_carlo.h`, `parallel_mc.h`, `vina.h` |
| `mutate_conf` | `model::mutate` (`mutate.cpp`) | `mutate.cpp` |
| `island_lga`, `tournament`, `crossover` | *(no Vina counterpart — follows AutoDock 4's LGA)* | — |
| `Shared`, `System`, `Docking`, `DockResult` | `Vina`, `model`, `igrid`, `scoring_function`, `conf` as passed around by `Vina::dock` | `vina.h`, `vina.cpp` |

**Licence and attribution.** AutoDock Vina is Copyright (c) 2006–2010, The
Scripps Research Institute, and is distributed under the Apache License 2.0; it
is the reference for every structure in the table above. AutoDock 4 is Copyright
(c) 1989–2007, The Scripps Research Institute, distributed under the GPL, and is
the reference for the AD4.2 force field, the AD4 typing and the Lamarckian
genetic algorithm. Meeko (Copyright (c) Forli Lab, Scripps Research; LGPL-2.1)
is the behavioural reference for ligand preparation. **No AutoDockTools (ADT /
MGLTools) code was read, borrowed or copied.**

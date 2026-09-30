// SPDX-License-Identifier: GPL-3.0-or-later
//! Structure input/output.
//!
//! The AutoDock PDBQT dialect is the native interchange format of the whole
//! AutoDock family. It is a strict superset of PDB: the last two columns carry
//! the AutoDock atom type and column 71-76 carries the Gasteiger partial
//! charge. Ligands additionally carry the rotatable-bond topology as nested
//! `ROOT` / `BRANCH a b` / `ENDBRANCH a b` records, which is exactly the
//! kinematic tree this crate consumes.
//!
//! The reader was written from the format specification and from the
//! *behaviour* documented by Meeko and AutoDock Vina; no AutoDockTools source
//! was consulted.

pub mod pdbqt;

pub use pdbqt::{
    parse_ligand_pdbqt, parse_receptor_pdbqt, write_pose_pdbqt, LigandRecord, ParseIssue,
    ParsedLigand, ReceptorRecord,
};

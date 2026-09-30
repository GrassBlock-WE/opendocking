// SPDX-License-Identifier: GPL-3.0-or-later
//! Error types for `dock-core`.

use std::fmt;

/// Result alias used throughout the crate.
pub type Result<T> = std::result::Result<T, DockError>;

/// Every recoverable failure mode of the docking kernel.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DockError {
    /// PDBQT (or any structure file) could not be parsed.
    Parse {
        /// 1-based line number.
        line: usize,
        /// Human readable explanation.
        message: String,
    },
    /// A structure was parsed but is not usable for docking.
    Invalid(String),
    /// A molecule contains no atoms.
    EmptyMolecule(&'static str),
    /// The ligand lies (partly) outside the search box.
    OutsideGrid(String),
    /// A requested map / grid is not initialised.
    NotInitialised(&'static str),
    /// Generic I/O failure.
    Io(String),
    /// A scoring-function-specific configuration problem.
    Scoring(String),
    /// The GPU backend is unavailable or failed.
    Gpu(String),
}

impl fmt::Display for DockError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DockError::Parse { line, message } => {
                write!(f, "parse error on line {line}: {message}")
            }
            DockError::Invalid(m) => write!(f, "invalid input: {m}"),
            DockError::EmptyMolecule(what) => write!(f, "{what} contains no atoms"),
            DockError::OutsideGrid(m) => write!(f, "ligand is outside the grid box: {m}"),
            DockError::NotInitialised(what) => write!(f, "{what} has not been initialised"),
            DockError::Io(m) => write!(f, "io error: {m}"),
            DockError::Scoring(m) => write!(f, "scoring error: {m}"),
            DockError::Gpu(m) => write!(f, "gpu error: {m}"),
        }
    }
}

impl std::error::Error for DockError {}

impl From<std::io::Error> for DockError {
    fn from(e: std::io::Error) -> Self {
        DockError::Io(e.to_string())
    }
}

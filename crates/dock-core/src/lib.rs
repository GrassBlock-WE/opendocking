// SPDX-License-Identifier: GPL-3.0-or-later
//! # dock-core — the OpenDocking high-performance docking kernel
//!
//! `dock-core` is a pure-Rust re-implementation of the classical AutoDock family
//! docking kernels (AutoDock 4 / AutoDock Vina / AutoDock-GPU), designed from the
//! published algorithms and from the freely licensed reference implementations:
//!
//! * AutoDock Vina — Copyright (c) 2006-2010, The Scripps Research Institute,
//!   Apache License 2.0. Used as the reference for the Vina/Vinardo empirical
//!   scoring terms, the analytic-derivative formulation, the iterated-local-search
//!   protocol and the PDBQT topology conventions.
//! * AutoDock 4 — Copyright (c) 1989-2007, The Scripps Research Institute,
//!   GPL. Used as the reference for the AD4 atom typing / parameter tables
//!   (`atom_constants.h`) and the AD4.2 pairwise force-field form.
//!
//! Every ported numerical constant keeps an attribution comment at its
//! definition site. See `docs/SCORING.md` for the full derivation.
//!
//! ## Layout
//!
//! | module | responsibility |
//! |---|---|
//! | [`math`] | quaternion / rigid-body primitives shared by every other module |
//! | [`rng`] | deterministic, serialisable RNG (reproducible docking runs) |
//! | [`atom`] | atom typing (AD4 + X-Score) and the parameter tables |
//! | [`molecule`] | atoms, bonds, graph-based bond perception, pair generation |
//! | [`kinematics`] | rigid clusters + torsion tree + forward kinematics |
//! | [`scoring`] | scoring functions, analytic gradients, affinity grids |
//! | [`search`] | BFGS and Solis-Wets local search, Monte-Carlo ILS, island GA |
//! | [`cancel`] | cooperative pause / resume / abort token polled by the searches |
//! | [`io`] | PDBQT reader/writer |
//! | [`gpu`] | optional `wgpu` compute backend (grids, batch kinematics, batch pairs) |
//! | [`docking`] | the high-level orchestration object (the "Vina" equivalent) |
//!
//! All modules forbid `unsafe` code; the only exception in the whole workspace
//! would be explicitly annotated SIMD intrinsics, of which there are currently
//! none (LLVM auto-vectorises the hot loops).

#![deny(unsafe_code)]
#![warn(missing_debug_implementations)]
// `math::PI` is deliberately spelled as the literal AutoDock Vina uses
// (`common.h`) instead of `std::f64::consts::PI`: the kernel has to stay
// bit-compatible with the reference implementation, so the constant is written
// out on purpose. `clippy::approx_constant` is deny-by-default and would
// otherwise reject the entire crate (it is not a new warning: the constant and
// its rationale predate this change).
#![allow(clippy::approx_constant)]

pub mod atom;
pub mod cancel;
pub mod docking;
pub mod error;
pub mod gpu;
pub mod io;
pub mod kinematics;
pub mod math;
pub mod molecule;
pub mod rng;
pub mod scoring;
pub mod search;

pub use atom::{AdType, Element, XsType};
pub use cancel::{CancelState, CancelToken};
pub use docking::{
    dock_batch, BatchJob, BatchOutcome, DockOptions, DockResult, Docking, Shared, System,
};
pub use error::{DockError, Result};
pub use kinematics::{Conf, DofLayout, KinematicTree, MovableModel};
pub use math::{DMat3, DQuat, DVec3};
pub use molecule::{Atom, Bond, Molecule, Pair};
pub use rng::Rng;
pub use scoring::grid::{AffinityGrid, GridBox};
pub use scoring::{ScoreComponents, ScoringFunction, SfChoice, Weights};
pub use search::{Caps, EnergyModel, LocalSearch, Pose, SearchParams, SolisWets, SolisWetsParams};

/// Crate version, surfaced to the Python layer and the CLI.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

// SPDX-License-Identifier: GPL-3.0-or-later
//! Regression tests against a real, flexible ligand.
//!
//! Erlotinib (29 heavy atoms, 11 torsions, two nested 2-methoxyethoxy chains)
//! is the smallest ligand that exercises *nested* `BRANCH` records. A flat
//! single-torsion ligand such as benzamidine does not: the topology is deeper,
//! and a mistake in the frame bookkeeping shows up as atoms displaced by the
//! distance between two frame origins.
//!
//! The fixtures are the actual files `odock prepare` produced from PDB 1M17.

use dock_core::io::pdbqt::{parse_ligand_pdbqt, parse_receptor_pdbqt};
use dock_core::kinematics::{build_ligand_tree, Conf, LigandConf};
use dock_core::math::DVec3;
use dock_core::molecule::{assign_types, perceive_bonds};
use dock_core::scoring::{make_scoring_function, SfChoice};

const LIGAND: &str = include_str!("data/erlotinib.pdbqt");
const RECEPTOR: &str = include_str!("data/egfr.pdbqt");

/// Applying the identity conformation must reproduce the input structure
/// exactly — that is the whole point of forward kinematics.
#[test]
fn identity_conformation_reproduces_a_nested_ligand() {
    let lig = parse_ligand_pdbqt(LIGAND).expect("the fixture must parse");
    let coords: Vec<DVec3> = lig.atoms.iter().map(|a| a.coords).collect();
    let (mut tree, root_idx) = build_ligand_tree(&lig.top, &coords, coords.len());
    assert_eq!(tree.num_torsions, 11, "the fixture declares 11 torsions");

    let mut out = coords.clone();
    let conf = LigandConf::null(coords[root_idx], tree.num_torsions);
    tree.apply_ligand(&conf, &mut out);

    let worst = out
        .iter()
        .zip(&coords)
        .map(|(a, b)| (*a - *b).length())
        .fold(0.0_f64, f64::max);
    assert!(
        worst < 1e-9,
        "forward kinematics displaced an atom by {worst:.4} A"
    );
}

/// Every movable atom must be owned by exactly one frame.
#[test]
fn frames_partition_the_movable_atoms() {
    let lig = parse_ligand_pdbqt(LIGAND).expect("the fixture must parse");
    let coords: Vec<DVec3> = lig.atoms.iter().map(|a| a.coords).collect();
    let n = coords.len();
    let (tree, _root) = build_ligand_tree(&lig.top, &coords, n);

    let mut owner = vec![usize::MAX; n];
    fn walk(node: &dock_core::kinematics::Node, owner: &mut [usize], id: usize) {
        for i in node.atoms.0..node.atoms.1 {
            assert_eq!(owner[i], usize::MAX, "atom {i} is owned twice");
            owner[i] = id;
        }
        for c in &node.children {
            walk(c, owner, id);
        }
    }
    walk(&tree.root, &mut owner, 0);
    let orphans: Vec<usize> = (0..n).filter(|&i| owner[i] == usize::MAX).collect();
    assert!(
        orphans.is_empty(),
        "{} atoms belong to no frame: {orphans:?}",
        orphans.len()
    );
}

/// Rotating a torsion must be an isometry *within* every frame: atoms that are
/// rigidly connected keep their relative geometry, and every covalent bond
/// keeps its length. Distances across a rotated bond are of course free to
/// change — that is what a torsion is for.
#[test]
fn nested_torsions_preserve_intra_frame_geometry() {
    use dock_core::kinematics::Node;

    let lig = parse_ligand_pdbqt(LIGAND).unwrap();
    let coords: Vec<DVec3> = lig.atoms.iter().map(|a| a.coords).collect();
    let n = coords.len();
    let (mut tree, root_idx) = build_ligand_tree(&lig.top, &coords, n);

    let conf = LigandConf {
        position: coords[root_idx] + DVec3::new(4.0, -2.0, 1.5),
        orientation: dock_core::DQuat::from_axis_angle(DVec3::new(1.0, 2.0, -1.0).normalize(), 0.9),
        torsions: (0..tree.num_torsions).map(|k| 0.3 * (k as f64) - 1.0).collect(),
    };
    let mut out = coords.clone();
    tree.apply_ligand(&conf, &mut out);

    // 1. Every frame is a rigid body.
    fn check(node: &Node, coords: &[DVec3], out: &[DVec3]) {
        for i in node.atoms.0..node.atoms.1 {
            for j in (i + 1)..node.atoms.1 {
                let before = (coords[i] - coords[j]).length();
                let after = (out[i] - out[j]).length();
                assert!(
                    (before - after).abs() < 1e-9,
                    "frame atoms {i}/{j}: {before:.6} -> {after:.6}"
                );
            }
        }
        for c in &node.children {
            check(c, coords, out);
        }
    }
    check(&tree.root, &coords, &out);

    // 2. Every covalent bond keeps its length.
    let mut bonded = parse_ligand_pdbqt(LIGAND).unwrap().atoms;
    perceive_bonds(&mut bonded);
    for (i, a) in bonded.iter().enumerate() {
        for b in &a.bonds {
            let j = b.other;
            if j <= i {
                continue;
            }
            let before = (coords[i] - coords[j]).length();
            let after = (out[i] - out[j]).length();
            assert!(
                (before - after).abs() < 1e-9,
                "bond {i}-{j}: {before:.6} -> {after:.6}"
            );
        }
    }
}

/// A real-system check of the exact scorer against a brute-force pair sum.
///
/// This is the test that catches a mistyped receptor, a missing cell-list
/// neighbour or a wrongly skipped atom: any of them changes the total.
#[test]
fn exact_scorer_matches_a_brute_force_sum_on_a_real_receptor() {
    let mut ligand_atoms = parse_ligand_pdbqt(LIGAND).unwrap().atoms;
    perceive_bonds(&mut ligand_atoms);
    assign_types(&mut ligand_atoms);

    let mut receptor = parse_receptor_pdbqt(RECEPTOR).unwrap().atoms;
    perceive_bonds(&mut receptor);
    assign_types(&mut receptor);

    let sf = make_scoring_function(SfChoice::Vina, None);
    let dims = [dock_core::scoring::grid::GridDim::empty(); 3];
    let nc = dock_core::scoring::noncache::NonCache::new(
        receptor.clone(),
        sf.as_ref(),
        dims,
        1e6,
    );

    let coords: Vec<DVec3> = ligand_atoms.iter().map(|a| a.coords).collect();
    let got = nc.eval(
        sf.as_ref(),
        &ligand_atoms,
        &coords,
        1000.0,
        false,
        coords.len(),
        None,
    );

    let cutoff = sf.cutoff();
    let mut expect = 0.0;
    for a in &ligand_atoms {
        if a.xs == dock_core::atom::XsType::W {
            continue;
        }
        let mut this_e = 0.0;
        for b in &receptor {
            if b.xs == dock_core::atom::XsType::W {
                continue;
            }
            let r = (a.coords - b.coords).length();
            if r < cutoff {
                this_e += sf.pair_energy(a, b, r);
            }
        }
        dock_core::math::curl_scalar(&mut this_e, 1000.0);
        expect += this_e;
    }
    assert!(
        (got - expect).abs() < 1e-9,
        "exact scorer gave {got:.6}, brute force {expect:.6}"
    );
}

/// The same system must also score identically through the public facade.
#[test]
fn system_energy_matches_the_exact_scorer() {
    use dock_core::docking::{build_system, DockOptions, System};
    use dock_core::search::{Caps, EnergyModel};
    use dock_core::scoring::grid::GridBox;

    let extent = {
        let mut lo = DVec3::splat(f64::MAX);
        let mut hi = DVec3::splat(f64::MIN);
        let lig = parse_ligand_pdbqt(LIGAND).unwrap();
        for a in &lig.atoms {
            lo = lo.min(a.coords);
            hi = hi.max(a.coords);
        }
        (hi - lo) + DVec3::splat(16.0)
    };
    let centre = {
        let lig = parse_ligand_pdbqt(LIGAND).unwrap();
        lig.atoms.iter().map(|a| a.coords).sum::<DVec3>() / lig.atoms.len() as f64
    };
    let options = DockOptions {
        use_grid: false,
        refine: true,
        box_: GridBox::new(centre, extent, 0.5),
        ..Default::default()
    };
    let mut system: System = build_system(RECEPTOR, LIGAND, &options).unwrap();
    system.force_exact = true;
    let conf: Conf = system.movable.initial_conf();
    let energy = system.eval(&conf, Caps::authentic());
    let brute = {
        let mut ligand = parse_ligand_pdbqt(LIGAND).unwrap().atoms;
        perceive_bonds(&mut ligand);
        assign_types(&mut ligand);
        let mut receptor = parse_receptor_pdbqt(RECEPTOR).unwrap().atoms;
        perceive_bonds(&mut receptor);
        assign_types(&mut receptor);
        let sf = make_scoring_function(SfChoice::Vina, None);
        let mut total = 0.0;
        for a in &ligand {
            if a.xs == dock_core::atom::XsType::W {
                continue;
            }
            let mut this_e = 0.0;
            for b in &receptor {
                if b.xs == dock_core::atom::XsType::W {
                    continue;
                }
                let r = (a.coords - b.coords).length();
                if r < sf.cutoff() {
                    this_e += sf.pair_energy(a, b, r);
                }
            }
            dock_core::math::curl_scalar(&mut this_e, 1000.0);
            total += this_e;
        }
        total
    };
    // `eval` returns the raw conformational energy; with a rigid receptor the
    // ligand's internal energy cancels against `unbound`, so it equals the
    // intermolecular sum.
    assert!(
        (energy - brute).abs() < 1e-6,
        "system gave {energy:.6}, brute force {brute:.6}"
    );
}

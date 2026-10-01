# SPDX-License-Identifier: GPL-3.0-or-later
"""Measure what an ensemble of receptor conformations changes, on real structures.

Run it::

    python examples/ensemble_validation.py            # the whole report
    python examples/ensemble_validation.py --fast     # one seed, fewer ligands

Everything it prints is *measured* on the bundled crystal structures (or on a
second conformation fetched into ``tests/data`` with ``odock fetch``) and is the
source of the numbers quoted in ``docs/ENSEMBLE.md``.  It writes the same
numbers as JSON to ``out/ensemble_validation.json`` so a reader can diff them
against the documentation.

The three questions it answers:

1. How far apart are the conformations?  Sequence identity, the binding-site
   RMSD after a site fit, the whole-protein CA RMSD, and the per-residue
   displacement of the site (results 1).
2. What does the ensemble do to one ligand?  The best affinity per conformation,
   which conformation produced the winning pose, how much the affinity moves on a
   receptor swap, and the robustness score (results 2 and 3).
3. Does the ensemble change the *ranking*, compared with any single structure?
   The same library docked into each conformation alone and into the ensemble
   (result 4).

What it does not answer, and cannot: whether the ligand selects a conformation
or induces one.  Docking scores a fixed receptor; it cannot tell those apart.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))

from odock import ensemble as ens  # noqa: E402
from odock.prepare import box_from_points  # noqa: E402

DATA = ROOT / "tests" / "data"

#: Genuine multi-conformation cases, all in ``tests/data``.
#: ``(label, [(file, provenance)], site residue)``.
CASES = {
    "er-alpha": (
        "3ERT (4-hydroxytamoxifen, antagonist) vs 1ERE chain A (estradiol, agonist)",
        [(DATA / "3ERT.pdb", None), (DATA / "1ERE_A.pdb", None)],
        "OHT",
    ),
    "hiv-protease": (
        "1HVR (XK263) vs 1HXW (A-84538): two HIV-1 protease isolates, 97 % identity",
        [(DATA / "1HVR.pdb", None), (DATA / "1HXW.pdb", None)],
        "XK2",
    ),
    "trypsin": (
        "3PTB (benzamidine, pH 8.0) vs 2PTN (ligand-free, 1.55 A): the control",
        [(DATA / "3PTB.pdb", None), (DATA / "2PTN.pdb", None)],
        "BEN",
    ),
}

#: A small library of real ligands for the ranking question.
LIBRARY = {
    "benzamidine": "N=C(N)c1ccccc1",
    "estradiol": str(DATA / "EST.sdf"),
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "naphthol": "c1ccc2c(c1)ccc(c2)O",
    "caffeine": "Cn1cnc2c1c(=O)n(C)c(=O)n2C",
}


def _box(conformations, residue, buffer: float = 6.0):
    ligand = ens.ligand_coords(conformations[0], residue)
    return box_from_points(ligand, buffer=buffer)


def report_alignment(cases) -> list:
    """Result 1: how far apart the conformations are, in numbers."""
    rows = []
    for key, (title, files, residue) in cases.items():
        paths = [path for path, _ in files]
        if not all(path.exists() for path in paths):
            print(f"!! {key}: missing {paths}")
            continue
        conformations = ens.read_conformations(paths)
        box = _box(conformations, residue)
        aligned = ens.align_conformations(
            conformations, box=box, site_radius=8.0, max_site_residues=30
        )
        entry = {
            "case": key,
            "title": title,
            "reference": aligned.labels[0],
            "site": aligned.site,
            "conformations": [
                {
                    "label": alignment.label,
                    "residues": conformation.n_residues(),
                    "identity": alignment.identity,
                    "overlap": alignment.overlap,
                    "matched_residues": alignment.matched_residues,
                    "site_residues": alignment.n_site_residues,
                    "site_rmsd": alignment.site_rmsd,
                    "global_ca_rmsd": alignment.global_rmsd,
                    "max_displacement": alignment.max_displacement,
                    "mean_displacement": alignment.mean_displacement,
                    "worst": sorted(
                        alignment.displacement.items(), key=lambda kv: -kv[1]
                    )[:5],
                }
                for conformation, alignment in zip(aligned.conformations, aligned.alignments)
            ],
        }
        rows.append(entry)
        print(f"== {key}: {title}")
        print(aligned.table())
        print()
    return rows


def build(case: str, outdir: Path):
    title, files, residue = CASES[case]
    paths = [path for path, _ in files]
    conformations = ens.read_conformations(paths)
    box = _box(conformations, residue)
    built = ens.build_ensemble(
        paths, box=box, outdir=outdir / case, site_radius=8.0, max_site_residues=30,
    )
    return built, box


def report_one_ligand(case: str, ligand, seeds, exhaustiveness: int, num_poses: int):
    """Results 2 and 3: one ligand against the ensemble, over several seeds."""
    built, box = build(case, Path("out/ensemble_validation"))
    entries = []
    for seed in seeds:
        started = time.perf_counter()
        result = ens.dock_ensemble(
            ligand, built, exhaustiveness=exhaustiveness, num_poses=num_poses, seed=seed
        )
        rescoring = ens.cross_rescore(result, built)
        score = ens.robustness_score(result, rescoring)
        per = result.best_per_conformation()
        entry = {
            "seed": seed,
            "per_conformation": per,
            "winner": result.winning_conformation(),
            "gap": (
                None
                if len(per) < 2
                else max(per.values()) - min(per.values())
            ),
            "best_overall": result.best_affinity,
            "n_poses": len(result.poses),
            "n_clusters": len(result.clusters),
            "top_cluster_size": result.clusters[0].size if result.clusters else 0,
            "top_cluster_conformations": (
                result.clusters[0].conformations if result.clusters else []
            ),
            "robustness": score.as_dict(),
            "seconds": time.perf_counter() - started,
        }
        entries.append(entry)
        print(
            f"  seed {seed}: winner {entry['winner']}, gap {entry['gap']:.3f} kcal/mol, "
            f"robustness {score.score:.3f} "
            f"(support {score.n_supporting}/{score.n_conformations}, "
            f"movement {score.movement:.3f}), {score.n_scored} rescore(s), "
            f"{entry['seconds']:.1f} s"
        )
    return entries


#: Which residue of each structure carries the co-crystallised ligand, so the
#: box (and the region of interest) is always derived in the *reference* frame.
SITE_RESIDUES = {
    "er-alpha": ("3ERT.pdb", "OHT", "1ERE_A.pdb", "EST"),
    "hiv-protease": ("1HVR.pdb", "XK2", "1HXW.pdb", "RIT"),
    "trypsin": ("3PTB.pdb", "BEN", "2PTN.pdb", None),
}

#: Pocket-analysis settings per case, so this script reproduces the tables in
#: ``docs/POCKETS.md`` exactly.  The trypsin control uses a tighter region and a
#: smaller pocket cap because its site is smaller; the other two share theirs.
POCKET_SETTINGS = {
    "er-alpha": {"region": 14.0, "max_pockets": 12},
    "hiv-protease": {"region": 14.0, "max_pockets": 12},
    "trypsin": {"region": 12.0, "max_pockets": 10},
}


def report_pockets(
    cases, *, reference_index: int = 0, region_override: float | None = None
) -> list:
    """Result 5: which cavities exist in some conformations and not others.

    Both reference directions are measured where each structure has its own
    ligand: swapping which structure anchors the frame is what turns "this cavity
    closes" into "this cavity is absent from the reference", and the two are not
    the same statement.
    """
    from odock import pockets as pocket_ensemble

    rows = []
    for key, (title, files, _site) in cases.items():
        paths = [path for path, _ in files]
        if not all(path.exists() for path in paths):
            print(f"!! {key}: missing {paths}")
            continue
        ligand_file, ligand_name, other_file, other_name = SITE_RESIDUES[key]
        frame_file = [ligand_file, other_file][reference_index]
        frame_ligand = [ligand_name, other_name][reference_index]
        if frame_ligand is None:
            # A ligand-free structure cannot define the site itself: the box comes
            # from its near-identical sibling, which is the point of the control.
            frame_file, frame_ligand = ligand_file, ligand_name
            print(
                f"   note: the ligand-free structure is the reference; the box comes "
                f"from {frame_file} ({frame_ligand}), whose site differs by the RMSD "
                "in section 1"
            )
        conformations = ens.read_conformations(paths)
        frame_path = DATA / frame_file
        frame_index = paths.index(frame_path) if frame_path in paths else 0
        ligand = ens.ligand_coords(conformations[frame_index], frame_ligand)
        from odock.prepare import box_from_points

        box = box_from_points(ligand, buffer=6.0)
        aligned = ens.align_conformations(
            conformations, reference=int(reference_index), box=box,
            site_radius=8.0, max_site_residues=30,
        )
        comparison = pocket_ensemble.compute_comparison(
            aligned, box=box, min_volume=50.0,
            region_radius=(
                POCKET_SETTINGS[key]["region"]
                if region_override is None
                else float(region_override)
            ),
            max_pockets=int(POCKET_SETTINGS[key]["max_pockets"]),
        )
        entry = {
            "case": key,
            "title": title,
            "reference": comparison.reference,
            "labels": comparison.labels,
            "noise": {label: floor.as_dict() for label, floor in comparison.noise.items()},
            "noise_volume": comparison.noise_volume,
            "n_pockets": {label: len(pockets) for label, pockets in comparison.detections.items()},
            "tracks": [track.as_dict() for track in comparison.tracks],
            "cryptic": [track.index for track in comparison.cryptic()],
            "seconds": comparison.elapsed,
        }
        rows.append(entry)
        print(f"== {key} (reference {comparison.reference}): {title}")
        print(f"   {comparison.noise_line()}")
        print(comparison.table(limit=12))
        candidates = comparison.cryptic()
        print(f"   cryptic candidates: {len(candidates)}")
        for track in candidates:
            print(
                f"     track {track.index + 1}: {track.labels_present} "
                f"V {track.volume_min:.0f}-{track.volume_max:.0f} A^3, local free "
                f"{track.local_free_min:.0f}-{track.local_free_max:.0f} A^3, "
                f"stability {track.stability:.2f}, lining J "
                f"{track.lining_jaccard():.2f}, score {track.cryptic_score:.2f}"
            )
            print(f"       lining: {', '.join(track.lining_reference()[:8])}")
            print(f"       trace: {track.openness_trace()}")
        print()
    return rows


def report_generated(*, outdir: Path, combinations: int = 16, max_rotamers: int = 5,
                     min_rmsd: float = 0.3, seed: int = 7) -> dict:
    """Result 6: an ensemble generated in silico from one structure.

    Generates from 3PTB (the only bundled structure with a co-crystallised ligand
    *and* a validated experimental partner, 2PTN), reports the sampling table, the
    site spread against the experimental pairs, the cost, and then runs the pocket
    analysis on the generated set -- because the question that matters is not
    "does it generate something" but "does the analysis reach the same verdicts".
    """
    from odock import generate as generator
    from odock import pockets as pocket_ensemble

    source = DATA / "3PTB.pdb"
    if not source.exists():
        print(f"!! missing {source}")
        return {}
    conformation = ens.read_conformations([source])[0]
    box = box_from_points(ens.ligand_coords(conformation, "BEN"), buffer=6.0)
    started = time.perf_counter()
    generated = generator.generate_ensemble(
        conformation, box=box, site_radius=6.0, max_rotamers=max_rotamers,
        min_rmsd=min_rmsd, combinations=combinations, seed=seed, source=source,
    )
    build_seconds = time.perf_counter() - started
    print("== generated ensemble (3PTB, the trypsin S1 site) ==")
    print(generated.text(limit=20))

    written = generated.write(outdir)
    storage = sum(path.stat().st_size for path in written)
    print()
    print(
        f"cost: {build_seconds:.2f} s to build {generated.n_conformations} "
        f"conformation(s) ({build_seconds / max(1, generated.n_conformations):.3f} s "
        f"each), {storage / 1024:.0f} KB on disk "
        f"({storage / max(1, generated.n_conformations) / 1024:.0f} KB per member)"
    )

    # The analysis on the generated set, in the frame it already shares.
    aligned = ens.align_conformations(
        [ens.read_conformations([path])[0] for path in written],
        reference=0, box=box, site_radius=8.0, max_site_residues=30, superpose=False,
    )
    comparison = pocket_ensemble.compute_comparison(
        aligned, box=box, region_radius=12.0, min_volume=50.0, max_pockets=10,
    )
    print()
    print("== the pocket analysis on the generated ensemble ==")
    print(f"   {comparison.noise_line().splitlines()[0][:160]}...")
    print(comparison.table(limit=8))
    print(f"   cryptic candidates: {len(comparison.cryptic())}")
    for track in comparison.ranked()[:4]:
        print(
            f"     track {track.index + 1}: found {track.n_found}/"
            f"{track.n_conformations}, free volume {track.local_free_min:.0f}-"
            f"{track.local_free_max:.0f} A^3, closure fraction "
            f"{track.closure_fraction:.2f}, cryptic={track.cryptic}"
        )

    return {
        "conformations": generated.n_conformations,
        "residues": [entry.as_dict() for entry in generated.residues],
        "spread": generated.spread,
        "comparison": generated.comparison(),
        "cost": {
            "build_seconds": build_seconds,
            "seconds_per_conformation": build_seconds / max(1, generated.n_conformations),
            "bytes_total": storage,
            "bytes_per_member": storage / max(1, generated.n_conformations),
        },
        "pockets": {
            "n_tracks": len(comparison.tracks),
            "n_cryptic": len(comparison.cryptic()),
            "noise_volume": comparison.noise_volume,
            "tracks": [track.as_dict() for track in comparison.tracks],
        },
    }


def report_waters(*, library: Path, exhaustiveness: int = 2, num_poses: int = 2,
                  seed: int = 42) -> dict:
    """Result 7: the water network, the displaced waters, and the pose correlation.

    First the network and the conserved/displaced classification on the one
    experimental pair in the repository that differs in water content *and* ligand
    content (3PTB carries benzamidine, 2PTN does not) -- the offline check the
    mandate asks for.  Then the per-pose displacement count against affinity across
    a small library docked into 3PTB, reported with n and a bootstrap interval, and
    reported as unresolvable when n is too small to read.
    """
    from odock import waters as water_analysis

    first, second = DATA / "3PTB.pdb", DATA / "2PTN.pdb"
    if not first.exists() or not second.exists():
        print(f"!! missing {first} or {second}")
        return {}
    conformations = ens.read_conformations([first, second], keep_water=True)
    box = box_from_points(ens.ligand_coords(conformations[0], "BEN"), buffer=6.0)
    ens.align_conformations(conformations, box=box, superpose=True)
    analysis = water_analysis.compare_water_sites(conformations)
    print("== water network and conserved sites (3PTB vs 2PTN) ==")
    print(analysis.text(limit=14))

    from odock.cli import _read_poses
    from odock.prepare import prepare_receptor
    from odock.screen import ScreenConfig, screen_ligands

    outdir = ROOT / "out" / "water_screen"
    outdir.mkdir(parents=True, exist_ok=True)
    # The campaign docks into the holo structure with benzamidine stripped:
    # leaving the co-crystallised ligand in would occupy the very site being
    # docked into, and the waters are not needed for the pose count (it is a
    # geometric check against the conserved sites measured above).
    prepared = outdir / "receptor.pdbqt"
    prepare_receptor(str(first), prepared, strip=["BEN"], keep_water=False)
    config = ScreenConfig(
        receptors=[str(prepared)], inputs=[str(library)], box=box, outdir=outdir,
        exhaustiveness=int(exhaustiveness), num_poses=int(num_poses),
        interactions=False, filters=False, progress=False, seed=int(seed), resume=False,
    )
    summary = screen_ligands(config)
    rows = []
    for record in summary.records:
        if record.status != "ok" or not record.pose_file:
            continue
        pose_file = outdir / record.pose_file
        if not pose_file.exists():
            continue
        poses = _read_poses(pose_file)
        counted = water_analysis.pose_displacement(poses, analysis.conserved, radius=3.5)
        if not counted:
            continue
        best = min(counted, key=lambda entry: entry["affinity"])
        rows.append(
            {
                "ligand": record.name,
                "n_poses": len(counted),
                "affinity": float(record.affinity),
                "displaced_any": max(entry["displaced"] for entry in counted),
                "displaced_best": best["displaced"],
                "mean_displaced": float(np.mean([entry["displaced"] for entry in counted])),
            }
        )
    print()
    print("== per-pose displacement of conserved waters vs affinity ==")
    print(f"   {len(rows)} ligand(s) with poses")
    for row in rows:
        print(
            f"   {row['ligand']:<28} affinity {row['affinity']:7.3f}  "
            f"displaced (best pose) {row['displaced_best']}  "
            f"max over poses {row['displaced_any']}  mean {row['mean_displaced']:.2f}"
        )
    correlation_best = water_analysis.correlate(
        [row["displaced_best"] for row in rows], [row["affinity"] for row in rows]
    )
    correlation_mean = water_analysis.correlate(
        [row["mean_displaced"] for row in rows], [row["affinity"] for row in rows]
    )
    print()
    for name, result in (("best pose", correlation_best), ("mean over poses", correlation_mean)):
        print(
            f"   correlation ({name}): rho {result['rho']:+.3f} "
            f"[{result['low']:+.3f}, {result['high']:+.3f}] n={result['n']} "
            f"{result['note']}"
        )
    return {
        "summary": analysis.summary(),
        "displaced": [site.as_dict() for site in analysis.displaced],
        "pose_rows": rows,
        "correlation": correlation_best,
        "correlation_mean": correlation_mean,
    }


def report_ranking(case: str, library, exhaustiveness: int, num_poses: int, seed: int):
    """Result 4: does the ensemble change which ligand ranks first?"""
    built, box = build(case, Path("out/ensemble_validation"))
    from odock.prepare import prepare_ligand

    per_ligand = {}
    for name, source in library.items():
        result = ens.dock_ensemble(
            source, built, exhaustiveness=exhaustiveness, num_poses=num_poses,
            seed=seed, ligand_name=name,
        )
        rescoring = ens.cross_rescore(result, built)
        score = ens.robustness_score(result, rescoring)
        per_ligand[name] = {
            "per_conformation": result.best_per_conformation(),
            "ensemble_best": result.best_affinity,
            "winner": result.winning_conformation(),
            "robustness": score.as_dict(),
            "n_poses": len(result.poses),
        }
        print(
            f"  {name:<12} "
            + ", ".join(
                f"{label} {value:.3f}" for label, value in result.best_per_conformation().items()
            )
            + f" | ensemble {result.best_affinity:.3f} | robustness {score.score:.3f}"
        )
    del prepare_ligand

    single = {
        label: sorted(
            (
                (values["per_conformation"].get(label), name)
                for name, values in per_ligand.items()
            ),
            key=lambda item: (item[0] is None, item[0]),
        )
        for label in built.labels
    }
    combined = sorted(
        ((values["ensemble_best"], name) for name, values in per_ligand.items()),
        key=lambda item: (item[0] is None, item[0]),
    )
    out = {
        "case": case,
        "seed": seed,
        "per_ligand": per_ligand,
        "single_conformation_ranking": {
            label: [name for _value, name in order] for label, order in single.items()
        },
        "ensemble_ranking": [name for _value, name in combined],
        "reference_ranking": {
            label: [name for _value, name in order] for label, order in single.items()
        },
    }
    print()
    for label, order in single.items():
        print(f"  {label:<10} top-1 {order[0][1]} ({order[0][0]:.3f})")
    print(f"  ensemble   top-1 {combined[0][1]} ({combined[0][0]:.3f})")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fast", action="store_true", help="one seed, two ligands")
    parser.add_argument(
        "--pockets-only", action="store_true",
        help="only the pocket (cryptic/transient) section",
    )
    parser.add_argument(
        "--generate-only", action="store_true",
        help="only the generated-ensemble section",
    )
    parser.add_argument(
        "--waters-only", action="store_true",
        help="only the water network / displaced waters / pose-correlation section",
    )
    parser.add_argument(
        "--combinations", type=int, default=16,
        help="simultaneous rotamer changes in the generated ensemble",
    )
    parser.add_argument("--max-rotamers", type=int, default=5)
    parser.add_argument(
        "--generated-min-rmsd", type=float, default=0.3,
        help="pruning threshold of the generated ensemble (Å)",
    )
    parser.add_argument("--exhaustiveness", type=int, default=4)
    parser.add_argument("--num-poses", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7, help="seed for the generated ensemble")
    parser.add_argument("--case", default="er-alpha", choices=sorted(CASES))
    parser.add_argument(
        "--region", type=float, default=None,
        help="override the per-case region radius of the pocket analysis (Å); the "
             "defaults reproduce the tables in docs/POCKETS.md",
    )
    parser.add_argument("--json-out", default="out/ensemble_validation.json")
    args = parser.parse_args()

    seeds = [42] if args.fast else [42, 7, 2024]
    library = dict(list(LIBRARY.items())[: (2 if args.fast else len(LIBRARY))])
    report = {"generated": time.strftime("%Y-%m-%dT%H:%M:%S")}

    print("### 1. the conformations\n")
    report["alignment"] = report_alignment(CASES)

    if not args.pockets_only and not args.generate_only and not args.waters_only:
        title, files, residue = CASES[args.case]
        print(f"\n### 2/3. estradiol into {args.case}: {title}\n")
        report["one_ligand"] = {
            "case": args.case,
            "ligand": "estradiol",
            "exhaustiveness": args.exhaustiveness,
            "num_poses": args.num_poses,
            "runs": report_one_ligand(
                args.case, DATA / "EST.sdf", seeds, args.exhaustiveness, args.num_poses
            ),
        }

        print(f"\n### 4. the ranking, single structures against the ensemble ({args.case})\n")
        report["ranking"] = report_ranking(
            args.case, library, args.exhaustiveness, args.num_poses, seed=42
        )

    if not args.generate_only and not args.waters_only:
        print("\n### 5. cryptic and transient pockets\n")
        pocket_rows = []
        for reference_index in (0, 1):
            pocket_rows.extend(
                report_pockets(
                    CASES, reference_index=reference_index, region_override=args.region
                )
            )
        report["pockets"] = pocket_rows

    if not args.waters_only:
        print("\n### 6. an ensemble generated from one structure\n")
        report["generated"] = report_generated(
            outdir=ROOT / "out" / "generated_3ptb",
            combinations=args.combinations,
            max_rotamers=args.max_rotamers,
            min_rmsd=args.generated_min_rmsd,
            seed=args.seed,
        )

    print("\n### 7. waters: the network, the displaced ones, and the pose correlation\n")
    report["waters"] = report_waters(
        library=ROOT / "demo" / "library.smi",
        exhaustiveness=args.exhaustiveness,
        num_poses=max(2, args.num_poses // 2),
        seed=42,
    )

    target = Path(args.json_out)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    print(f"\nwrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

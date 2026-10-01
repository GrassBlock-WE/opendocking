# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for :mod:`odock.pockets`: cryptic and transient cavities across an ensemble.

Two layers again:

* **synthetic cavities** whose geometry is exact -- a spherical shell of atoms at
  a known radius has a known interior volume, a ray of 10 Å from its centre hits
  it in all 26 directions and a ray of 5 Å hits it in none -- so the two metrics
  this module defines can be checked against arithmetic rather than against
  themselves;
* **real crystal pairs**, where the measured answer is the one
  ``docs/POCKETS.md`` quotes: three cavities of 3ERT that 1ERE_A does not have
  (and two more tracks the report rejects, with the numbers that reject them),
  plus a trypsin control that must produce none.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from odock import ensemble as ens
from odock import pockets as pk


# ---------------------------------------------------------------------------
# Synthetic cavities
# ---------------------------------------------------------------------------


def shell_atoms(radius: float, n: int = 300, res_name: str = "ALA", res_id: int = 1):
    """`n` carbon atoms spread evenly on a sphere of `radius` Å.

    A Fibonacci lattice, so the shell has no holes: every direction from the
    centre meets an atom, which is what makes the buriedness arithmetic below
    exact.
    """
    index = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1.0 - 2.0 * index / n)
    theta = math.pi * (1.0 + 5.0 ** 0.5) * index
    points = radius * np.stack(
        [np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)], axis=1
    )
    return [
        ens.Atom(
            name=f"C{k}", element="C", res_name=res_name, res_id=res_id, chain="A",
            x=float(x), y=float(y), z=float(z),
        )
        for k, (x, y, z) in enumerate(points)
    ]


def filler_atoms(radius: float, res_name: str = "ALA", res_id: int = 1):
    """A 3-D lattice of atoms filling the shell, spaced `radius` apart.

    A ring is not enough: a ring leaves the poles open and a probe still reaches
    the middle, so a "filled" cavity would measure as partly open and the test
    would be measuring its own fixture.  A lattice closes the sphere in every
    direction, and it keeps the shell's residue identity so the two conformations
    line the cavity with the same residues.
    """
    atoms = []
    for i in (-1, 0, 1):
        for j in (-1, 0, 1):
            for k in (-1, 0, 1):
                atoms.append(
                    ens.Atom(
                        name=f"F{i + 1}{j + 1}{k + 1}", element="C", res_name=res_name,
                        res_id=res_id, chain="A",
                        x=float(i * radius), y=float(j * radius), z=float(k * radius),
                    )
                )
    return atoms


def conformation(label: str, atoms, path: Path | None = None) -> ens.Conformation:
    return ens.Conformation(label=label, atoms=list(atoms), records=[], path=path)


def test_the_shell_itself_is_the_reference_geometry():
    """The fixture's own arithmetic: 300 atoms on a sphere, all inside it."""
    atoms = shell_atoms(8.0, 300)
    coords = np.array([[a.x, a.y, a.z] for a in atoms])
    assert np.allclose(np.linalg.norm(coords, axis=1), 8.0)
    assert len(atoms) == 300


def test_local_free_volume_is_the_free_sphere_inside_the_shell():
    """A shell at 8 Å with carbon radii leaves a free ball of radius 8-3.1 Å."""
    atoms = shell_atoms(8.0)
    free = pk.local_free_volume((0.0, 0.0, 0.0), atoms, radius=6.0)
    expected = 4.0 / 3.0 * math.pi * (8.0 - 1.70 - 1.4) ** 3
    assert free == pytest.approx(expected, rel=0.15)
    # And an empty receptor has the whole sphere.
    assert pk.local_free_volume((0.0, 0.0, 0.0), [], radius=6.0) == pytest.approx(
        4.0 / 3.0 * math.pi * 6.0 ** 3
    )


def test_local_free_volume_collapses_when_the_cavity_is_filled():
    open_atoms = shell_atoms(8.0)
    filled = open_atoms + filler_atoms(2.5)
    before = pk.local_free_volume((0.0, 0.0, 0.0), open_atoms, radius=6.0)
    after = pk.local_free_volume((0.0, 0.0, 0.0), filled, radius=6.0)
    assert before > 300.0
    assert after < 0.25 * before


def test_buriedness_is_exact_for_a_closed_shell():
    """All 26 directions hit the shell at 8 Å when the ray is long enough."""
    atoms = shell_atoms(8.0)
    point = np.zeros((1, 3))
    assert pk.buriedness(point, atoms, ray_length=10.0) == pytest.approx(1.0)
    # A 5 Å ray cannot reach the shell: nothing is walled.
    assert pk.buriedness(point, atoms, ray_length=5.0) == pytest.approx(0.0)
    # An empty receptor has no walls at all.
    assert pk.buriedness(point, [], ray_length=10.0) == pytest.approx(0.0)


def test_buriedness_is_between_the_two_extremes_at_an_opening():
    """A shell with a hole: not fully buried, not open."""
    atoms = [a for a in shell_atoms(8.0) if not (a.x > 6.0 and abs(a.y) < 3.0 and abs(a.z) < 3.0)]
    value = pk.buriedness(np.zeros((1, 3)), atoms, ray_length=10.0)
    assert 0.0 < value < 1.0


def test_clearest_point_picks_the_point_farthest_from_atoms():
    points = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    atoms = [
        ens.Atom(name="C1", element="C", res_name="ALA", res_id=1, chain="A", x=0.0, y=0.0, z=0.1)
    ]
    assert pk.clearest_point(points, atoms) == pytest.approx((2.0, 0.0, 0.0))
    assert pk.clearest_point(np.zeros((0, 3)), atoms) is None


def test_the_detector_finds_the_synthetic_cavity():
    """The 9.8 Å-wide interior is cut into sub-pockets; their volumes add up.

    Measured on this fixture: five sub-pockets at 3.0 Å from the middle, 79-86 Å³
    each, buriedness 0.945-0.950, all lined by the shell's single residue.  The
    sum is an independent cross-check: this module's own free-volume measurement
    of the same cavity (485 Å³ at 1 Å spacing) must roughly match what the
    detector's sub-pockets account for.
    """
    atoms = shell_atoms(8.0)
    found = pk.detect_pockets(
        conformation("open", atoms), spacing=1.0, min_volume=50.0, max_pockets=12,
        ray_length=10.0, atoms=atoms,
    )
    assert found, "the detector should see the inside of a closed shell"
    assert len(found) >= 2, "a 9.8 Å cavity is cut into several sub-pockets"
    for observed in found:
        assert np.linalg.norm(np.asarray(observed.center)) < 4.0
        assert 60.0 < observed.volume < 120.0
        assert observed.buriedness > 0.9
        assert observed.lining == ["ALA1 A"]
        assert observed.local_free_volume > 300.0
    total = sum(observed.volume for observed in found)
    assert total == pytest.approx(
        pk.local_free_volume((0.0, 0.0, 0.0), atoms, radius=6.0), rel=0.25
    )


def test_matching_needs_both_distance_and_lining_residues():
    def observation(label: str, center, lining):
        return pk.PocketObservation(
            conformation=label, index=0, center=tuple(center), volume=100.0,
            score=1.0, n_points=100, points=np.zeros((0, 3)), lining=list(lining),
        )

    detections = {
        "A": [observation("A", (0.0, 0.0, 0.0), ["ASP1 A"])],
        "B": [
            observation("B", (1.0, 0.0, 0.0), ["ASP1 A"]),   # same place, same lining
            observation("B", (1.5, 0.0, 0.0), ["GLU2 A"]),   # same place, other lining
            observation("B", (9.0, 0.0, 0.0), ["ASP1 A"]),   # too far
        ],
    }
    tracks = pk.match_pockets(detections, ["A", "B"], reference="A")
    assert len(tracks) == 3
    matched = [track for track in tracks if track.n_found == 2]
    assert len(matched) == 1
    assert matched[0].observations["B"].lining == ["ASP1 A"]
    assert matched[0].lining_jaccard() == pytest.approx(1.0)


def test_a_synthetic_ensemble_reports_a_transient_cavity(tmp_path):
    """Open shell vs filled shell: one track, support 1/2, a large free-volume change."""
    from odock.prepare import BoxSpec

    open_atoms = shell_atoms(8.0)
    filled = open_atoms + filler_atoms(2.5)
    box = BoxSpec(center=(0.0, 0.0, 0.0), size=(40.0, 40.0, 40.0), spacing=1.0)
    comparison = pk.compute_comparison(
        [conformation("open", open_atoms), conformation("filled", filled)],
        box=box, spacing=1.0, min_volume=50.0, max_pockets=8, ray_length=10.0,
        noise=False, region_radius=12.0,
    )
    assert comparison.labels == ["open", "filled"]
    assert comparison.reference == "open"
    tracks = [track for track in comparison.tracks if track.n_found]
    assert tracks
    transient = [track for track in tracks if track.transient]
    assert transient, "the filled structure must not reveal the open cavity the same way"
    track = transient[0]
    assert track.labels_present == ["open"]
    # Measured on this fixture: 485 Å³ of free volume at the anchor when the
    # shell is empty, 0 Å³ when the lattice fills it.
    assert track.local_free["open"] > 300.0
    assert track.local_free["filled"] < 5.0
    assert track.local_free_change > 300.0
    assert track.closed_elsewhere
    assert track.changing
    assert "openness trace" in comparison.text()
    # Without a noise measurement the flags fall back to the change itself, and
    # the report says the resolution was not measured.
    assert any("not measured" in warning for warning in comparison.warnings)


def test_detector_noise_measures_the_same_structure_twice():
    atoms = shell_atoms(8.0)
    floor = pk.detector_noise(
        conformation("open", atoms),
        jitters=((1.3, 1.0), (1.4, 1.0), (1.5, 1.0)),
        min_volume=50.0, max_pockets=8, ray_length=10.0, atoms=atoms,
    )
    assert len(floor.settings) == 3
    assert floor.tracks >= 1
    assert math.isfinite(floor.absolute_change)
    assert floor.absolute_change >= 0.0
    assert "unchanged structure" in floor.text()


def test_pocket_atoms_strips_the_ligand_inside_the_box():
    """A holo structure's site is not a cavity while its inhibitor is in it."""
    from odock.prepare import BoxSpec

    ligand = [
        ens.Atom(name=f"L{k}", element="C", res_name="LIG", res_id=99, chain="A",
                 x=0.1 * k, y=0.0, z=0.0)
        for k in range(10)
    ]
    protein = shell_atoms(8.0)
    structure = conformation("holo", protein + ligand)
    box = BoxSpec(center=(0.0, 0.0, 0.0), size=(20.0, 20.0, 20.0), spacing=1.0)
    kept = pk.pocket_atoms(structure, box=box)
    assert all(atom.res_name != "LIG" for atom in kept)
    assert len(kept) == len(protein)
    # ... and it is kept when no box is given (nothing says it is in the site).
    assert any(atom.res_name == "LIG" for atom in pk.pocket_atoms(structure))


# ---------------------------------------------------------------------------
# Real crystal pairs
# ---------------------------------------------------------------------------


def er_alpha_pair(data_dir):
    first, second = data_dir / "3ERT.pdb", data_dir / "1ERE_A.pdb"
    if not first.exists() or not second.exists():
        pytest.skip("missing the bundled ERα structures")
    return first, second


def trypsin_pair(data_dir):
    first, second = data_dir / "3PTB.pdb", data_dir / "2PTN.pdb"
    if not first.exists() or not second.exists():
        pytest.skip("missing the bundled trypsin structures")
    return first, second


def build(data_dir, first, second, ligand, *, reference=0, region=14.0, noise=True, max_pockets=12):
    from odock.prepare import box_from_points

    conformations = ens.read_conformations([first, second])
    points = ens.ligand_coords(conformations[reference], ligand)
    box = box_from_points(points, buffer=6.0)
    aligned = ens.align_conformations(
        conformations, reference=reference, box=box, site_radius=8.0, max_site_residues=30
    )
    return pk.compute_comparison(
        aligned, box=box, region_radius=region, min_volume=50.0,
        max_pockets=max_pockets, noise=noise,
    )


def test_the_trypsin_control_candidate_is_detector_noise(data_dir):
    """3PTB vs 2PTN: sites 0.14 Å apart. Measured, and the answer is *no*.

    The detector does flag one cavity as present in 3PTB and absent in 2PTN, and
    saying so is the point of this test: the control's raw candidate is real, and
    the report must reject it for the right reason.  Measured here it moves the
    free volume by 40 Å³ (509 -> 549 Å³ at the same point, 7.6 %) against the
    detector's own resolution of 39 Å³ -- the same cavity, with the detector's
    sub-pocket partition dropping a marginal one in the other structure.  The
    ERα pair moves 5-11x its resolution; this moves 1.03x.
    """
    first, second = trypsin_pair(data_dir)
    comparison = build(data_dir, first, second, "BEN", region=12.0, noise=True, max_pockets=10)
    assert comparison.labels == ["3PTB", "2PTN"]
    assert comparison.tracks, "the control pair still has cavities"
    assert comparison.cryptic() == [], comparison.text()

    transient = [track for track in comparison.tracks if track.transient]
    assert transient, "the raw detector output does differ between the two structures"
    for track in transient:
        # The reason it is rejected, as a number: the openness change is below
        # the margin over the detector's own resolution.
        assert track.local_free_change < pk.CHANGE_MARGIN * track.local_noise
        assert not track.changing
        # And the two structures disagree about a cavity that is *there* in both:
        # the free volume at the same point is within a few percent.
        present = [track.local_free[label] for label in track.labels_present]
        absent = [track.local_free[label] for label in track.labels_absent]
        assert max(absent) > 0.85 * max(present)
    # The report explains the empty list rather than leaving it blank.
    text = comparison.text()
    assert "detector resolution" in text
    assert "(none)" in text or "no transient site" in text


@pytest.mark.slow
def test_the_estrogen_receptor_loses_four_cavities_on_agonist_binding(data_dir):
    """3ERT (antagonist, H12 open) vs 1ERE_A (agonist): the measured finding.

    With the stability check on, **three** cavities qualify: all three are present
    in 3ERT and gone in 1ERE_A (the helix-12 face of the ligand-binding domain
    closes over them).  A fourth track -- an agonist-only cavity -- is excluded
    here precisely because its presence is not reproduced under the jittered
    detector settings (stability 0.10); the reverse-direction test below measures
    that cavity with the check off.  Each qualifying track is lined by the *same*
    residues in both structures (Jaccard 1.00), so the structures show the same
    residues moved rather than two different pockets that happen to overlap, and
    each moves its free volume by 5-11x the detector's measured resolution.
    """
    first, second = er_alpha_pair(data_dir)
    comparison = build(data_dir, first, second, "OHT", region=14.0, noise=True, max_pockets=12)
    candidates = comparison.cryptic()
    assert len(candidates) >= 3, comparison.text()
    assert all(track.lining_jaccard() == pytest.approx(1.0) for track in candidates)
    assert all(track.stability >= 0.6 for track in candidates)
    assert all(
        track.local_free_change > pk.CHANGE_MARGIN * track.local_noise
        for track in candidates
    )
    assert all(track.transient for track in candidates)
    assert all(track.labels_present == ["3ERT"] for track in candidates)

    # The biggest cavity of the antagonist structure is filled in the agonist one.
    biggest = max(candidates, key=lambda track: track.volume_max)
    assert biggest.volume_max > 400.0
    assert biggest.local_free["3ERT"] > 300.0
    assert biggest.local_free["1ERE_A"] < 50.0
    # The detector's own volume noise is reported and is large: the report must
    # not present the volume as if it were reproducible to a few Å³.
    assert comparison.noise_volume > 50.0
    assert "reading the volumes" in comparison.text()
    # The candidate lining includes the helix-12 face of the domain.
    lining = set(biggest.lining_reference())
    assert {"TRP383 A", "GLU380 A"} & lining
    assert "THR347 A" in lining and "ASP351 A" in lining


def test_the_reference_direction_changes_what_absent_means(data_dir):
    """With 1ERE_A as the reference, 3ERT's cavities become cryptic sites.

    Measured: the biggest signal in this direction is a cavity the agonist
    structure does not reveal at all -- 9 Å³ of free volume at the anchor in
    1ERE_A against 360 Å³ in 3ERT, a 351 Å³ opening, 175x the local resolution.
    Its detector volume is only 50 Å³, right at ``--min-volume``, so the verdict
    rests on the *collapse of the free volume* rather than on the volume.
    """
    first, second = er_alpha_pair(data_dir)
    comparison = build(
        data_dir, first, second, "EST", reference=1, region=14.0, noise=False,
        max_pockets=12,
    )
    assert comparison.reference == "1ERE_A"
    candidates = comparison.cryptic()
    assert candidates, comparison.text()
    absent = [track for track in candidates if track.absent_from_reference]
    assert absent, comparison.text()
    assert all(track.labels_present == ["3ERT"] for track in absent)
    assert all(track.n_found == 1 for track in absent)
    biggest = max(absent, key=lambda track: track.local_free_change)
    assert biggest.local_free["1ERE_A"] < 20.0
    assert biggest.local_free["3ERT"] > 300.0
    assert biggest.local_free_change > 300.0
    assert biggest.closed_elsewhere


def test_the_pocket_pdb_puts_every_cavity_point_in_the_common_frame(data_dir):
    first, second = trypsin_pair(data_dir)
    comparison = build(data_dir, first, second, "BEN", region=12.0, noise=False, max_pockets=6)
    text = comparison.pocket_pdb()
    lines = [line for line in text.splitlines() if line.startswith("HETATM")]
    assert lines, "no cavity points were written"
    assert text.startswith("REMARK  ODOCK ENSEMBLE POCKETS")
    assert text.rstrip().endswith("END")
    for line in lines:
        assert len(line) >= 54
        float(line[30:38])  # the coordinates parse
    # The residue name is the conformation and the chain is the cavity.
    resnames = {line[17:20].strip() for line in lines}
    assert resnames <= {"3PT", "2PT"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_ensemble_pockets_documents_itself(capsys):
    from odock.cli import main

    with pytest.raises(SystemExit) as excinfo:
        main(["ensemble", "pockets", "--help"])
    assert excinfo.value.code == 0
    text = capsys.readouterr().out
    for flag in (
        "--grid-spacing", "--probe", "--min-volume", "--max-pockets", "--buriedness",
        "--lining-radius", "--local-radius", "--match-radius", "--min-overlap",
        "--region-radius", "--no-noise", "--pockets-pdb", "--json-out", "--box-ligand",
    ):
        assert flag in text, flag
    assert "cryptic" in text


def test_cli_ensemble_pockets_runs_on_the_control_pair(data_dir, tmp_path, capsys):
    from odock.cli import main

    first, second = trypsin_pair(data_dir)
    target = tmp_path / "pockets.json"
    pdb = tmp_path / "pockets.pdb"
    code = main(
        [
            "ensemble", "pockets",
            "-r", str(first), str(second),
            "--box-ligand", "BEN", "--buffer", "6",
            "--region-radius", "12", "--max-pockets", "8",
            "--no-noise",
            "--pockets-pdb", str(pdb),
            "--json-out", str(target),
            "-q",
        ]
    )
    assert code == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["labels"] == ["3PTB", "2PTN"]
    assert payload["reference"] == "3PTB"
    assert payload["tracks"], "the control pair still has cavities"
    assert payload["parameters"]["min_volume"] == 50.0
    assert payload["n_pockets"]["3PTB"] > 0
    assert payload["noise"] == {}  # --no-noise
    assert pdb.read_text(encoding="utf-8").count("HETATM") > 0


def test_cli_ensemble_pockets_reports_a_missing_site(data_dir, tmp_path, capsys):
    from odock.cli import main

    first, _second = trypsin_pair(data_dir)
    code = main(
        [
            "ensemble", "pockets",
            "-r", str(first), str(tmp_path / "missing.pdb"),
            "--box-ligand", "BEN",
            "-q",
        ]
    )
    assert code == 2
    assert "no such file" in capsys.readouterr().err

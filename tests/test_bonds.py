# SPDX-License-Identifier: GPL-3.0-or-later
"""Bond perception: the right atoms, and only the right atoms.

The ligand's reference answer is RDKit's ``DetermineConnectivity`` where it is
installed.  The receptor is checked against trypsin's own chemistry — 219
peptide bonds for 220 numbered residues, six disulfides, no other cross-residue
contact — and against the same RDKit oracle.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from odock.gui import bonds as B
from odock.gui.structure import Atom, parse_pdbqt

ROOT = Path(__file__).resolve().parents[1]
LIGAND_PDBQT = ROOT / "demo" / "3ptb" / "ligand.pdbqt"
RECEPTOR_PDB = ROOT / "tests" / "data" / "3PTB.pdb"
RECEPTOR_PDBQT = ROOT / "demo" / "3ptb" / "receptor.pdbqt"

#: HETATM records that are not part of the protein chain.
_NOT_PROTEIN = frozenset({"HOH", "BEN", "CA"})


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _atoms(path: Path):
    return parse_pdbqt(path.read_text(encoding="utf-8", errors="replace"))[0].atoms


def _pairs(bonds):
    return set(B.bond_pairs(bonds))


def _degree(count, pairs):
    degree = [0] * count
    for i, j in pairs:
        degree[i] += 1
        degree[j] += 1
    return degree


def _distance(atoms, i, j):
    return math.dist((atoms[i].x, atoms[i].y, atoms[i].z),
                     (atoms[j].x, atoms[j].y, atoms[j].z))


def _atom(name, element, x, y, z, res_name="LIG", res_id=1, chain="A"):
    return Atom(name=name, element=element, res_name=res_name, res_id=res_id,
                chain=chain, x=float(x), y=float(y), z=float(z))


def _protein_groups(atoms):
    """``{(chain, res_id): [atom indices]}`` in file order, protein only."""
    groups = {}
    for index, atom in enumerate(atoms):
        if atom.res_name in _NOT_PROTEIN:
            continue
        groups.setdefault((atom.chain, atom.res_id), []).append(index)
    return groups


@pytest.fixture(scope="module")
def ligand():
    return _atoms(LIGAND_PDBQT)


@pytest.fixture(scope="module")
def receptor():
    return _atoms(RECEPTOR_PDB)


@pytest.fixture(scope="module")
def shipped_receptor():
    """The receptor PDBQT the workbench actually loads (polar hydrogens kept)."""
    return _atoms(RECEPTOR_PDBQT)


# ---------------------------------------------------------------------------
# 1. the ligand: which atom is joined to which
# ---------------------------------------------------------------------------


def test_benzamidine_bonds_join_the_right_atoms(ligand):
    atoms = ligand
    pairs = _pairs(B.perceive_bonds(atoms, kind="ligand"))

    # 6 aromatic C-C ring bonds + the C-C bond to the amidine carbon + 2 C-N.
    heavy = {
        ("C1", "C2"), ("C2", "C3"), ("C3", "C4"),
        ("C4", "C5"), ("C5", "C6"), ("C6", "C1"),
        ("C1", "C"), ("C", "N1"), ("C", "N2"),
    }
    # every hydrogen is on a nitrogen of the amidine group
    hydrogen = {("N1", "H10"), ("N1", "H11"), ("N2", "H12"), ("N2", "H13")}
    got = {frozenset((atoms[i].name, atoms[j].name)) for i, j in pairs}
    assert got == {frozenset(pair) for pair in heavy | hydrogen}
    assert len(pairs) == 13

    ring = [name for name in ("C1", "C2", "C3", "C4", "C5", "C6")]
    ring_bonds = [p for p in got if p <= set(ring)]
    assert len(ring_bonds) == 6
    assert frozenset(("C1", "C")) in got
    assert frozenset(("C2", "C")) not in got

    # The two amidine nitrogens are 2.27 A apart and must never be joined.
    assert frozenset(("N1", "N2")) not in got

    # Every hydrogen has degree exactly 1, and only to its own nitrogen.
    degree = _degree(len(atoms), pairs)
    for index, atom in enumerate(atoms):
        if atom.element == "H":
            assert degree[index] == 1
    assert {frozenset((atoms[i].name, atoms[j].name)) for i, j in pairs
            if atoms[i].element == "H" or atoms[j].element == "H"} == \
        {frozenset(pair) for pair in hydrogen}

    # No N...H contact is bonded: N1/N2 are joined to their own two H only.
    for index, atom in enumerate(atoms):
        if atom.element != "N":
            continue
        neighbours = {atoms[j if i == index else i].name
                      for i, j in pairs if index in (i, j)}
        assert len(neighbours) == 3
        assert neighbours == ({"C", "H10", "H11"} if atom.name == "N1"
                              else {"C", "H12", "H13"})


# ---------------------------------------------------------------------------
# 2. a hydrogen bond is not a bond
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("donor_acceptor", [2.60, 1.90])
def test_hydrogen_bond_contact_is_not_a_bond(donor_acceptor):
    """N-H...O: the hydrogen stays on its own nitrogen, whatever the distance.

    2.60 A is the distance the requirement names; 1.90 A is a *typical* (strong)
    hydrogen bond, so it checks that the tolerance is not quietly generous.
    """
    atoms = [
        _atom("N", "N", 0.0, 0.0, 0.0),
        _atom("H", "H", 1.01, 0.0, 0.0),
        _atom("O", "O", 1.01 + donor_acceptor, 0.0, 0.0),
    ]
    bonds = B.perceive_bonds(atoms, kind="ligand")
    assert _pairs(bonds) == {(0, 1)}
    assert B.bond_report(atoms, bonds)["over_valent"] == []

    # Distance mode is the naive behaviour this replaces: it wires the O to the
    # hydrogen (and, at 1.9 A, to the nitrogen as well).
    naive = _pairs(B.perceive_bonds(
        atoms, kind="ligand", connect_mode=B.CONNECT_MODE_DISTANCE))
    assert (1, 2) in naive
    assert ((0, 2) in naive) == (donor_acceptor < 2.59)


# ---------------------------------------------------------------------------
# 3. a methyl group
# ---------------------------------------------------------------------------


def test_methyl_hydrogens_bond_only_to_their_carbon():
    scale = 1.09 / math.sqrt(3.0)
    directions = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    atoms = [_atom("C", "C", 0.0, 0.0, 0.0)]
    for number, direction in enumerate(directions):
        atoms.append(_atom(f"H{number + 1}", "H", *(scale * c for c in direction)))

    # The three hydrogens of a methyl: every one bonded to C, none to each other.
    methyl = _pairs(B.perceive_bonds(atoms[:4], kind="ligand"))
    assert methyl == {(0, 1), (0, 2), (0, 3)}
    assert math.isclose(_distance(atoms, 1, 2), 1.78, abs_tol=0.02)
    assert (1, 2) not in methyl and (1, 3) not in methyl and (2, 3) not in methyl

    # ... and the fourth hydrogen completes methane, because C may take four.
    methane = _pairs(B.perceive_bonds(atoms, kind="ligand"))
    assert methane == {(0, 1), (0, 2), (0, 3), (0, 4)}
    for first in range(1, 5):
        for second in range(first + 1, 5):
            assert (first, second) not in methane


# ---------------------------------------------------------------------------
# 4/5. the receptor: peptide bonds, disulfides, valence
# ---------------------------------------------------------------------------


def test_receptor_cross_residue_bonds_are_peptide_or_disulfide(receptor):
    atoms = receptor
    pairs = _pairs(B.perceive_bonds(atoms, kind="receptor"))
    peptides = 0
    disulfides = 0
    offenders = []
    for i, j in sorted(pairs):
        first, second = atoms[i], atoms[j]
        if (first.chain, first.res_id) == (second.chain, second.res_id):
            continue
        length = _distance(atoms, i, j)
        symbols = {first.element, second.element}
        names = {first.name.upper(), second.name.upper()}
        if symbols == {"S"}:
            assert length <= 2.2
            disulfides += 1
        elif (symbols == {"C", "N"} and first.chain == second.chain
              and names == {"C", "N"} and length <= 1.6):
            peptides += 1
        else:
            offenders.append(
                (first.name, first.res_name, first.res_id,
                 second.name, second.res_name, second.res_id, length)
            )
    assert offenders == []
    # Trypsin has six disulfides; a wrong pairing would show up as a wrong count.
    assert disulfides == 6
    # 220 numbered residues in one chain -> 219 peptide bonds, +/- a couple.
    residues = _protein_groups(atoms)
    assert len(residues) == 220
    assert abs(peptides - (len(residues) - 1)) <= 2
    assert peptides == 219


def test_receptor_loses_no_peptide_bond_at_a_numbering_gap(receptor):
    """Every consecutive residue pair must be joined by its backbone C-N.

    3PTB's numbering skips a number six times (missing loops) and uses insertion
    codes three times.  All nine links are real 1.29-1.36 A peptide bonds in the
    coordinates, so a rule based on consecutive residue *numbers* alone would
    silently lose six of them.
    """
    atoms = receptor
    pairs = _pairs(B.perceive_bonds(atoms, kind="receptor"))
    groups = _protein_groups(atoms)
    ids = list(groups)
    expected = set()
    for first, second in zip(ids, ids[1:]):
        nitrogens = [i for i in groups[second] if atoms[i].name == "N"]
        for carbon in (i for i in groups[first] if atoms[i].name == "C"):
            for nitrogen in nitrogens:
                if _distance(atoms, carbon, nitrogen) <= 1.6:
                    expected.add((carbon, nitrogen))
    # every expected peptide bond is present ...
    assert expected <= pairs
    # ... and nothing extra was invented for the same criterion.
    perceived = {(i, j) for i, j in pairs
                 if {atoms[i].element, atoms[j].element} == {"C", "N"}
                 and {atoms[i].name, atoms[j].name} == {"C", "N"}
                 and (atoms[i].chain, atoms[i].res_id) != (atoms[j].chain, atoms[j].res_id)}
    assert perceived == expected
    assert len(expected) == 219

    # the six missing numbers, spelled out
    gaps = [(("ASN", 34), ("SER", 37)), (("LEU", 67), ("GLY", 69)),
            (("THR", 125), ("SER", 127)), (("SER", 130), ("ALA", 132)),
            (("LYS", 204), ("LEU", 209)), (("SER", 217), ("GLY", 219))]
    for (res_a, id_a), (res_b, id_b) in gaps:
        carbon = next(i for i in groups[(atoms[0].chain, id_a)]
                      if atoms[i].res_name == res_a and atoms[i].name == "C")
        nitrogen = next(i for i in groups[(atoms[0].chain, id_b)]
                        if atoms[i].res_name == res_b and atoms[i].name == "N")
        assert tuple(sorted((carbon, nitrogen))) in pairs


@pytest.mark.parametrize("which", ["ligand", "receptor", "shipped"])
def test_no_element_exceeds_its_valence(which, ligand, receptor, shipped_receptor):
    atoms, kind = {
        "ligand": (ligand, "ligand"),
        "receptor": (receptor, "receptor"),
        "shipped": (shipped_receptor, "receptor"),
    }[which]
    pairs = _pairs(B.perceive_bonds(atoms, kind=kind))
    degree = _degree(len(atoms), pairs)

    for index, atom in enumerate(atoms):
        assert degree[index] <= B.expected_valence(atom.element), (
            f"{which}:{atom.res_name}{atom.res_id}:{atom.name} has "
            f"{degree[index]} bonds"
        )
        if atom.element == "H":
            assert degree[index] == 1
    # Two hydrogens are never joined, and a metal or noble gas never is either.
    assert not any(atoms[i].element == atoms[j].element == "H" for i, j in pairs)
    assert not any(not B.is_bondable(atoms[i].element)
                   or not B.is_bondable(atoms[j].element) for i, j in pairs)
    assert B.bond_report(atoms, B.perceive_bonds(atoms, kind=kind))["over_valent"] == []


def test_valence_table_and_metal_handling():
    assert {e: B.expected_valence(e) for e in
            ("H", "C", "N", "O", "S", "P", "F", "Cl", "Br", "I")} == {
        "H": 1, "C": 4, "N": 4, "O": 2, "S": 6, "P": 5,
        "F": 1, "Cl": 1, "Br": 1, "I": 1,
    }
    for metal in ("Fe", "Zn", "Mg", "Mn", "Ca", "Na", "K"):
        assert B.expected_valence(metal) == 0
        assert B.is_bondable(metal) is False
    assert B.is_bondable("Xe") is False and B.is_bondable("He") is False
    assert B.is_bondable("C") is True

    # A ferrous ion 2.0 A from a cysteine sulfur stays a sphere: the ion is not
    # wired to its coordination sphere, though the sulfur keeps its C-S bond.
    atoms = [_atom("FE", "Fe", 0.0, 0.0, 0.0),
             _atom("SG", "S", 2.0, 0.0, 0.0, res_name="CYS"),
             _atom("CB", "C", 3.82, 0.0, 0.0, res_name="CYS")]
    pairs = _pairs(B.perceive_bonds(atoms, kind="ligand"))
    assert pairs == {(1, 2)}
    assert (0, 1) not in pairs


def test_valence_override_caps_an_element(ligand):
    """``valence=`` replaces single entries of the built-in table."""
    capped = _pairs(B.perceive_bonds(ligand, valence={"C": 2}))
    degree = _degree(len(ligand), capped)
    assert max(degree[i] for i, a in enumerate(ligand) if a.element == "C") <= 2
    assert len(capped) == 12 < 13          # the C1-C bond is the one dropped
    assert frozenset(("C1", "C")) not in {
        frozenset((ligand[i].name, ligand[j].name)) for i, j in capped
    }
    # A zero override means "use the built-in table" (PyMOL's default setting).
    assert _pairs(B.perceive_bonds(ligand, valence={"C": 0})) == \
        _pairs(B.perceive_bonds(ligand))
    assert _pairs(B.perceive_bonds(ligand, valence={"C": None})) == \
        _pairs(B.perceive_bonds(ligand))


# ---------------------------------------------------------------------------
# 6. the oracle: RDKit's own connectivity
# ---------------------------------------------------------------------------


def test_agrees_with_rdkit_determine_connectivity(ligand, receptor):
    pytest.importorskip("rdkit")
    rdDetermineBonds = pytest.importorskip("rdkit.Chem.rdDetermineBonds")
    from rdkit import Chem  # noqa: PLC0415 - only needed inside this test

    def connectivity(atoms):
        editable = Chem.RWMol()
        conformer = Chem.Conformer(len(atoms))
        for index, atom in enumerate(atoms):
            editable.AddAtom(Chem.Atom(atom.element))
            conformer.SetAtomPosition(index, (atom.x, atom.y, atom.z))
        mol = editable.GetMol()
        mol.AddConformer(conformer)
        rdDetermineBonds.DetermineConnectivity(mol)
        return {tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
                for bond in mol.GetBonds()}

    # The correctness oracle for the ligand ...
    assert _pairs(B.perceive_bonds(ligand, kind="ligand")) == connectivity(ligand)
    # ... and for the whole receptor: no invented bond, none missed.
    #
    # Exception, and it is deliberate: RDKit's xyz2mol connectivity gives a
    # metal ion bonds to whatever sits within a covalent-radius reach of it
    # (3PTB's calcium picks up six), while this module refuses to bond metals at
    # all — a drawn "bond" to a Ca2+ is a coordination contact, not a covalent
    # bond, and drawing it is exactly the mistake the user asked us to avoid.
    # So the comparison excludes bonds that touch a metal.
    metals = {index for index, atom in enumerate(receptor)
              if B.expected_valence(atom.element) == 0}
    expected = {pair for pair in connectivity(receptor)
                if pair[0] not in metals and pair[1] not in metals}
    assert _pairs(B.perceive_bonds(receptor, kind="receptor")) == expected


# ---------------------------------------------------------------------------
# 7. distance mode, kept for comparison
# ---------------------------------------------------------------------------


def test_distance_mode_is_strictly_more_promiscuous(receptor):
    atoms = receptor
    automatic = _pairs(B.perceive_bonds(atoms, kind="receptor"))
    naive = _pairs(B.perceive_bonds(
        atoms, kind="receptor", connect_mode=B.CONNECT_MODE_DISTANCE))

    assert automatic < naive
    assert len(naive) > 2 * len(automatic)
    # The naive mode runs no valence table, so it over-coordinates atoms ...
    naive_report = B.bond_report(
        atoms, B.perceive_bonds(atoms, kind="receptor",
                                connect_mode=B.CONNECT_MODE_DISTANCE))
    assert naive_report["over_valent"]
    assert naive_report["max_degree"] > 6
    for entry in naive_report["over_valent"]:
        assert entry["degree"] > entry["expected"]
        assert entry["expected"] == B.expected_valence(entry["element"])
    # ... while automatic mode never does.
    assert B.bond_report(atoms, B.perceive_bonds(atoms, kind="receptor"))["over_valent"] == []


def test_connect_cutoff_controls_distance_mode(ligand):
    def naive(cutoff):
        return _pairs(B.perceive_bonds(
            ligand, kind="ligand", connect_mode=B.CONNECT_MODE_DISTANCE,
            connect_cutoff=cutoff))

    # 1.1 A keeps only the N-H bonds (1.03 A) ...
    assert naive(1.1) == {(7, 9), (7, 10), (8, 11), (8, 12)}
    # ... 1.4 A picks up the two C=N and the shortest C-C bonds ...
    middle = naive(1.40)
    assert (7, 9) in middle and (0, 1) in middle and (0, 6) in middle
    assert (2, 3) not in middle            # C3-C4 is 1.404 A, just past it
    # ... and the default is PyMOL's 3.6 A, which also wires the near misses.
    wide = naive(B.DEFAULT_CONNECT_CUTOFF)
    assert _pairs(B.perceive_bonds(ligand)) < wide
    assert len(wide) > 20
    assert B.DEFAULT_CONNECT_CUTOFF == 3.6
    assert B.CONNECT_MODE_AUTO == 3 and B.CONNECT_MODE_DISTANCE == 0


# ---------------------------------------------------------------------------
# 8. the diagnostics the bond-check dialog reads
# ---------------------------------------------------------------------------


def test_bond_report_fields_are_self_consistent(ligand, receptor):
    for atoms, kind in ((ligand, "ligand"), (receptor, "receptor")):
        bonds = B.perceive_bonds(atoms, kind=kind)
        report = B.bond_report(atoms, bonds)
        pairs = _pairs(bonds)

        assert report["n_atoms"] == len(atoms)
        assert report["n_bonds"] == len(pairs) == len(bonds)
        assert report["n_single"] + report["n_double"] + report["n_triple"] == len(pairs)
        assert all(bond.order >= 1 for bond in bonds)

        degree = _degree(len(atoms), pairs)
        histogram = {}
        for value in degree:
            histogram[value] = histogram.get(value, 0) + 1
        assert report["degree_histogram"] == dict(sorted(histogram.items()))
        assert sum(report["degree_histogram"].values()) == len(atoms)
        # `isolated` means "a bondable atom that ended up with no bond", which is
        # the diagnostic that matters (a broken molecule). A metal with no bonds
        # is correctly unbonded rather than isolated, so it is excluded here even
        # though the degree histogram counts it in the zero bucket.
        bondable_zero = sum(
            1
            for index, atom in enumerate(atoms)
            if degree[index] == 0 and B.is_bondable(atom.element)
        )
        assert report["degree_histogram"].get(0, 0) >= report["isolated"]
        assert report["isolated"] == bondable_zero
        assert report["isolated"] == len(report["isolated_atoms"])
        assert report["max_degree"] == max(report["degree_histogram"])
        assert sum(d * n for d, n in report["degree_histogram"].items()) == 2 * len(pairs)

        lengths = sorted(_distance(atoms, i, j) for i, j in pairs)
        assert report["mean_length"] == pytest.approx(sum(lengths) / len(lengths))
        assert report["median_length"] == pytest.approx(lengths[len(lengths) // 2])
        assert report["min_length"] == pytest.approx(lengths[0])

        longest = report["longest"]
        assert longest["length"] == pytest.approx(lengths[-1])
        assert 0 <= longest["a"] < len(atoms) and 0 <= longest["b"] < len(atoms)
        assert tuple(sorted((longest["a"], longest["b"]))) in pairs
        assert longest["elements"] == (atoms[longest["a"]].element,
                                       atoms[longest["b"]].element)
        assert report["over_valent"] == []


def test_bond_report_ignores_unusable_bond_entries(ligand):
    bonds = [(0, 1), (0, 1), (1, 0), (2, 2), (0, 999), B.Bond(3, 4)]
    report = B.bond_report(ligand, bonds)
    assert report["n_bonds"] == 2            # only (0, 1) and Bond(3, 4) survive
    assert report["longest"] is not None
    assert B.bond_report(ligand, [])["n_bonds"] == 0
    assert B.bond_report(ligand, [])["longest"] is None
    assert B.bond_report(ligand, [])["mean_length"] is None


# ---------------------------------------------------------------------------
# compatibility with the call sites that already exist
# ---------------------------------------------------------------------------


def test_guess_bonds_wrapper_and_bond_unpacking(ligand):
    legacy = B.guess_bonds(ligand)
    assert legacy == B.bond_pairs(B.perceive_bonds(ligand, kind="ligand"))
    assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in legacy)
    assert (0, 1) in legacy
    # the old factor / max_bonds arguments are accepted and ignored
    assert B.guess_bonds(ligand, 1.25, 4) == legacy

    # Bond unpacks as (a, b), which the picking and renderer code relies on.
    for bond in B.perceive_bonds(ligand):
        i, j = bond
        assert (i, j) == (bond.a, bond.b)
        assert i < j
    assert B.Bond(3, 1).as_tuple() == (3, 1)   # dataclass stays as given


def test_structure_guess_bonds_alias_still_works(ligand):
    from odock.gui import structure

    assert set(structure.guess_bonds(ligand)) == _pairs(B.perceive_bonds(ligand, kind="ligand"))


def test_coords_and_elements_overrides_drive_perception():
    atoms = [_atom("H1", "H", 0.0, 0.0, 0.0), _atom("O1", "O", 9.0, 9.0, 9.0)]
    bonds = B.perceive_bonds(atoms, coords=[(0.0, 0.0, 0.0), (0.0, 0.0, 0.96)],
                             elements=["H", "O"])
    assert _pairs(bonds) == {(0, 1)}
    # an override that disagrees with the atom count is a clear error
    with pytest.raises(ValueError):
        B.perceive_bonds(atoms, coords=[(0.0, 0.0, 0.0)], elements=["H", "O"])


def test_input_validation(ligand):
    with pytest.raises(ValueError):
        B.perceive_bonds(ligand, connect_mode=1)
    with pytest.raises(ValueError):
        B.perceive_bonds(ligand, connect_mode=B.CONNECT_MODE_DISTANCE, connect_cutoff=0)
    with pytest.raises(ValueError):
        B.perceive_bonds(ligand, valence={"C": "lots"})
    assert B.perceive_bonds([]) == []
    assert B.perceive_bonds(ligand[:1]) == []
    assert B.bond_report([], [])["n_bonds"] == 0


def test_shipped_receptor_loads_with_sane_connectivity(shipped_receptor):
    """The real GUI input: 3ptb/receptor.pdbqt, polar hydrogens and all."""
    atoms = shipped_receptor
    bonds = B.perceive_bonds(atoms, kind="receptor")
    report = B.bond_report(atoms, bonds)
    assert report["n_bonds"] > 2000
    assert report["over_valent"] == []
    assert report["isolated"] == 0
    assert report["max_degree"] in (3, 4)
    # every one of the 364 polar hydrogens hangs off exactly one heavy atom
    assert sum(1 for atom in atoms if atom.element == "H") == 364

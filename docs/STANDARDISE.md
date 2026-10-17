# Standardising a library

`odock screen` ranks whatever molecules it is given. A **real** vendor library does not
arrive as a set of molecules: it arrives with salts, counter-ions, mixtures, formal
charges, duplicates under two tautomers, stereocentres nobody specified, and a header row
that is not a molecule. Rank that file and you are ranking the salt, or one arbitrary
component of a mixture — which is a number nobody can defend.

`odock standardise` is the ordered, switchable recipe that turns such a file into a
standard library **and the audit that says what happened to every input record**.

```bash
odock standardise vendor_library.smi \
    -o clean.smi --audit audit.csv --json report.json \
    --charge-policy neutralise --stereo refuse \
    --heavy-atoms 8:60 --mw 120:700
```

The same thing from Python:

```python
from odock.standardise import PropertyLimits, StandardiseConfig, standardise

report = standardise("vendor_library.smi", config=StandardiseConfig(
    charge_policy="neutralise",
    stereo_policy="refuse",
    limits=PropertyLimits(heavy_atoms=(8, 60), molecular_weight=(120.0, 700.0)),
))
print(report.summary())          # in / kept / rejected / merged, and why
print(report.kept[0].smiles)     # the cleaned molecule
assert report.balanced           # in == kept + rejected + merged
```

The rest of this page is the honest version of what the recipe does and does not know.

---

## 1. What standardisation is not

This is the section to read before quoting any number that came out of a standardised
library.

* **It is not registration.** No registry was consulted. There is no InChIKey round-trip
  against ChEMBL, PubChem or a corporate database, no salt-form table, no parent-structure
  lookup. The InChIKey in the audit is computed locally by RDKit for grouping; it is not a
  claim that the molecule *is* the registry's record of the same name.
* **There is no tautomer enumeration beyond one rule.** Duplicates are collapsed using
  RDKit's `TautomerEnumerator` canonical tautomer as the key. A molecule whose tautomers
  the enumerator does not generate will not collapse with its partner, and no attempt is
  made to enumerate a tautomer ensemble, let alone score it.
* **No step decides which form is bioactive.** `--charge-policy neutralise` does not mean
  "the neutral form binds"; it means "produce the neutral form and record that you did".
  The charge step *asks* the shared predicate
  [`odock.protonation.can_represent_formal_charge`](../python/odock/protonation.py)
  whether the molecule carries the charges pH 7.4 implies, and **records the answer**
  rather than acting on it. On the bundled, hand-curated 17-molecule library *10 of the 17*
  do not carry the charges pH 7.4 implies (a neutral carboxylic acid is not the species
  pH 7.4 describes). That is the assumption the tool surfaces, not a defect it repairs.
* **A cleaned library ranks differently from the raw one.** De-salting, neutralising,
  collapsing a tautomer pair and dropping stereochemistry all change which molecules are
  in the file and what shape they are scored in, so the ranking changes with them. That is
  a reason to **report both** — standardise, screen the cleaned library, and keep the raw
  file and the audit next to the results so a reader can see what was moved.
* **Unsupported means refused, not guessed.** The cases this recipe cannot decide are
  named in [§6](#6-cases-this-recipe-refuses) and refused with a reason. A plausible answer
  for a metal complex is worth less than an honest refusal.

---

## 2. Quick start on the bundled data

```bash
python -m odock.standardise demo/libraries/library.smi -o out/clean.smi --audit out/audit.csv
```

The bundled library is already standard — it is hand-curated, neutral and clean — so the
run changes nothing, which is itself the check that the recipe does not rewrite a library
that is already right:

```text
standardised 17 record(s) from demo\libraries\library.smi
  kept 17, rejected 0, merged 0
  recipe 120d96d3d65f (parse, repair, desalt, charge, stereo, dedupe, filter)
  charge: 7 record(s) carry the charges pH 7.4 implies, 10 do not (recorded, not changed
  — that is the assumption, see docs/STANDARDISE.md), 0 unknown
```

and the file it writes carries its own provenance:

```text
# odock standardise 0.1.0 — recipe 120d96d3d65f73abcb0462912b73207830e4283a134dfe823ce09c34fa3dd4ab
# recipe: schema 1; steps parse, repair, desalt, charge, stereo, dedupe, filter
# policies: charge=keep stereo=keep largest_on_tie=refuse tautomer_dedupe=true
# rdkit 2026.03.6
# counts: in 17, kept 17, rejected 0, merged 0
# columns: SMILES<TAB>name
# standardisation is not registration: no registry was consulted and the InChIKeys in the
# audit are computed locally (docs/STANDARDISE.md)
N=C(N)c1ccccc1	benzamidine
CNC(=N)c1ccccc1	benzamidine_methyl
...
```

Two columns, deliberately: a `.smi` that carries a third one stops round-tripping its names
through the rest of the toolchain (`odock.chem.ligand.read_ligands` treats everything after
the first field as the name). The InChIKey and everything else the audit knows lives in the
CSV and the JSON.

---

## 3. The recipe, step by step

The order is part of the answer, so it is named, checked, recorded in the output header and
**hashed**. Every step is switchable (`--no-repair`, `--no-desalt`, …) and a step that is
off is reported as *skipped*, never treated as if it had run and found nothing.

| # | step | what it does | what it reports | switch |
|---|---|---|---|---|
| 1 | **parse** | reads the record and sanitises it | refusals, with the line (or SDF record) number and RDKit's reason | always on |
| 2 | **repair** | RDKit's documented `Cleanup` normalisations | the canonical SMILES before and after, when it changed | `--no-repair` |
| 3 | **desalt** | keeps the **largest organic** component | every removed fragment: SMILES, heavy atoms, charge | `--no-desalt` |
| 4 | **charge** | `keep`, `neutralise` (RDKit `Uncharger`) or `refuse_charged` | each atom whose formal charge moved; the pH 7.4 predicate's statement | `--no-charge`, `--charge-policy` |
| 5 | **stereo** | `keep`, `drop` or `refuse` | the centres that lost their assignment, or the reason for refusing | `--no-stereo`, `--stereo` |
| 6 | **dedupe** | canonical SMILES **and** a tautomer-insensitive key | which molecule a duplicate collapsed into, and whether it was exact or a tautomer | `--no-dedupe`, `--exact-dedupe` |
| 7 | **filter** | property windows: heavy atoms, MW, net charge, rotatable bonds | **every** violation, naming the property, the value and the window | `--no-filter`, the window flags |

### Why the order is not free

`STEP_CONSTRAINTS` in the module is both the documentation and the check. A configuration
that puts a step where it cannot be honest is refused at construction, naming the
constraint:

```text
ValueError: dedupe has to run after stereo: the dedupe key is computed on the structure
the later steps produced; a key taken before the stereo policy makes the enantiomers that
collapse under stereo='drop' look like two distinct molecules
```

That is not a hypothetical. Measured, with two enantiomers (`C[C@H](F)Cl` and
`C[C@@H](F)Cl`):

| stereo policy | kept | merged |
|---|---|---|
| `keep` | 2 | 0 |
| `drop` | 1 | 1 (`exact duplicate`) |

Take the dedupe key before the stereo policy and the `drop` run would keep both, i.e. the
policy would silently do nothing. The test suite asserts both rows.

### The steps in detail

**(a) parse — refuse rather than repair.** A record that RDKit cannot parse, or that parses
but cannot be sanitised (a five-valent carbon from a broken export), is **refused** with its
position and the reason. Nothing is repaired into something plausible. A table header such
as `SMILES ID` is not skipped either: it is a record that fails to parse, and it appears in
the audit at line 1. Silently dropping something that looks like a molecule is how a
library loses an entry.

**(b) repair — RDKit's normalisations, reported.** `rdMolStandardize.Cleanup` fixes the
common export artefacts (charge-separated nitro groups, drawing errors). The audit records
the canonical SMILES before and after *only when they differ*. This step also does **not**
touch metals: a sodium counter-ion is exactly what desalting removes.

**(c) desalt — the largest organic component.** Components are separated, and the largest
one containing carbon is kept. Every other component is reported with its SMILES, heavy-atom
count and charge. Two rules worth naming:

* a **tie** for the largest component is *ambiguous* and refused by default; `--on-tie first`
  accepts the documented alphabetical-by-SMILES tie-break. A 50/50 mixture of two
  equal-sized molecules is not something a rule should silently decide;
* the choice is **cross-checked against RDKit's own `FragmentParent`**, and a disagreement
  is recorded as an observation — so the rule this document describes and the rule RDKit
  implements are known to agree, and the day they stop agreeing the audit says so.

**(d) charge — the policy, and the assumption.** `keep` changes nothing; `neutralise` runs
RDKit's `Uncharger` and reports every atom whose formal charge moved;
`refuse_charged` rejects any record with a non-zero net formal charge. Whatever the policy,
the record is then put to
[`protonation.can_represent_formal_charge`](../python/odock/protonation.py) — the same
predicate the fragment work uses — and the answer is stored per molecule
(`charge_consistent`, `charge_statement`). For a neutralised carboxylate the audit says:

```text
carboxylate would carry -1 at pH 7.4 but its charges sum to +0.00 e: this charge set
cannot represent that formal charge, so any electrostatic term computed from it is computed
for a different species (docs/PROTONATION.md)
```

That is the honest outcome: the file is now neutral, and the tool has written down what it
did *not* model.

**(e) stereo — a policy, because an unspecified centre changes the score.** `keep` records
that a centre was left unspecified and leaves it; `drop` removes the assignments, naming the
atoms (and, as the table above shows, that changes the dedupe key); `refuse` rejects a record
with an unspecified centre, naming it:

```text
1 stereocentre(s) with no specified configuration (atom 2); an unspecified centre changes
the shape and therefore the score, so the record is refused rather than ranked under an
assumed configuration
```

Only tetrahedral centres are examined. E/Z double-bond stereo is not assessed, and that is a
limitation of this step rather than a silent assumption.

**(f) filter — every rejection, with its reason.** Windows are inclusive and any side may be
open (`8:`, `:500`). A rejection names the property, the value and the window, so it can be
counted mechanically:

```text
ethanol: filter — heavy_atoms 3 outside 8..40; molecular_weight 46.07 outside 120..400
```

The rotatable-bond count comes from `odock.chem.ligand.rotatable_bonds` — the perception the
kinematic tree is built from — so the window counts exactly the torsions the search will
sample, not a different notion of flexibility.

---

## 4. The audit

Every input record produces exactly one audit entry, whatever happened to it, and the
summary accounts for all of them. A deliberately vendor-shaped file shows every status:

```text
standardised 9 record(s) from <text>
  kept 3, rejected 4, merged 2
  recipe 120d96d3d65f (parse, repair, desalt, charge, stereo, dedupe, filter)
  rejected by category: invalid 2, unsupported 2
  merged: exact duplicate 1, tautomer 1
  changed: dedupe 2, desalt 1
```

and the per-record lines behind those numbers:

```text
smi line 1 ID: invalid — 'SMILES' is not a parsable SMILES
smi line 2 sodium_acetate: kept — kept the largest organic component CC(=O)[O-]
    (4 heavy atom(s), charge -1); removed [Na+] (1 heavy atom(s), charge +1)
smi line 3 hydroxy_pyridine: kept
smi line 4 pyridone: merged into hydroxy_pyridine (tautomer)
smi line 5 ethanol: kept
smi line 6 ethanol_again: merged into ethanol (exact duplicate)
smi line 7 broken: invalid — 'not_a_molecule' is not a parsable SMILES
smi line 8 ambiguous: unsupported — ambiguous largest component: 2 organic components of
    4 heavy atom(s) each (CCCC, CCCO); pass largest_on_tie='first' (or --on-tie first) to
    accept the alphabetical tie-break
smi line 9 methylmercury: unsupported — the standardised molecule still contains a metal;
    coordination chemistry is not modelled by this recipe (docs/STANDARDISE.md, Unsupported)
```

### The counting invariant

```text
in == kept + rejected + merged
```

`report.balanced` exposes it and `report.assert_balanced()` raises if it fails, naming the
counts. `standardise()` asserts it before returning, so **no record can vanish
unexplained** — a claim the test suite checks with a file that produces all three statuses,
and a hand-made report that cheats the invariant to prove the assertion fires.

### The files

| file | what it is |
|---|---|
| `clean.smi` | the kept library, with the provenance header above |
| `audit.csv` | one row per **input** record — kept, rejected and merged alike |
| `report.json` | the recipe, the counts, the aggregates and (optionally) every audit entry |

The CSV columns are stable and snake_case: `position`, `position_kind`, `name`, `status`,
`category`, `reason`, `input_smiles`, `output_smiles`, `merged_into`, `merged_into_name`,
`merge_kind`, `canonical_key`, `tautomer_key`, `inchikey`, `heavy_atoms`,
`molecular_weight`, `net_charge`, `rotatable_bonds`, `charge_consistent`, `changed`,
`removed_fragments`, `changes`, `observations`. `changes` and `observations` are separate
columns on purpose: "this molecule was desalted" and "this molecule was left with an
unspecified centre" are different kinds of fact.

The rejection categories are fixed, so a report can count them:

| category | meaning |
|---|---|
| `invalid` | the record is not a molecule (unparsable, unsanitisable, empty) |
| `unsupported` | this recipe does not decide this case ([§6](#6-cases-this-recipe-refuses)) |
| `stereo` | refused by `--stereo refuse` |
| `charge` | refused by `--charge-policy refuse_charged` |
| `filter` | outside a requested property window |

---

## 5. Determinism, the recipe hash, and the header

The same input and the same configuration produce **byte-identical** output. There is no
timestamp, no path, no hostname and no dictionary-order dependence anywhere in the text; the
test suite asserts it for both the `.smi` and the CSV.

The recipe hash covers **every setting and the schema version**, and deliberately excludes
paths, the machine and any timestamp — the same reasoning as
[`Protocol.hash`](../python/odock/protocol.py): two labs that recorded the same decisions
compute the same digest, and the digest proves *identity, not correctness*. Measured:

| recipe | short hash |
|---|---|
| defaults (`charge=keep`, `stereo=keep`) | `120d96d3d65f` |
| `--charge-policy neutralise` | `f4241a46f7fb` |
| `--charge-policy refuse_charged` | `c4ff600e0379` |
| `--stereo drop` | `d86a4c5139fa` |
| `--on-tie first` | `38bd925c9bc8` |
| `--exact-dedupe` | `603b88ffb104` |
| `--no-repair` | `858525f11378` |

A consequence worth noticing: `--charge-policy neutralise` on the *already neutral* bundled
library changes no molecule, and still produces a different hash and a different header —
because the recipe, not the result, is what the header identifies.

The header also records the RDKit version, because a canonical SMILES is RDKit's answer and
a different RDKit can write a different string. That is the traceability the header is for.

---

## 6. Cases this recipe refuses

An unsupported input that is **named** is worth more than a plausible answer. Each of these
is refused, with the reason in the audit:

| case | why it is refused | the audit says |
|---|---|---|
| a metal that survives every step | coordination chemistry is not modelled; the largest-component rule would be a guess | `the standardised molecule still contains a metal; coordination chemistry is not modelled by this recipe` |
| two organic components of the same size | the largest-component rule has no answer; the tie is real | `ambiguous largest component: N organic components of M heavy atom(s) each (…)` |
| more than `--max-fragments` components (default 8) | a mixture, not a salt | `it looks like a mixture, and choosing a component would be a guess` |
| no component contains carbon | there is no ligand to keep | `no component is organic (…), so there is no ligand to keep` |
| a record that parses but will not sanitise | the structure is ambiguous; repairing it would invent chemistry | `the structure cannot be sanitised (…); it is refused rather than repaired` |
| an unspecified stereocentre under `--stereo refuse` | the shape, and therefore the score, is unknown | `N stereocentre(s) with no specified configuration (atom X)` |
| `.mol2`, `.pdb`, `.pdbqt`, `.mol` input | these formats carry no reliable bond orders, which is exactly what the parse step judges | `is not a supported standardisation input … convert it to SMILES or SDF first` |
| `--no-desalt` with a metal counter-ion present | the counter-ion cannot be removed, so the molecule still contains a metal | `still contains a metal … note that --no-desalt was given` |

Two notes on scope:

* a **metal counter-ion is not a refusal** — it is removed by desalting and reported like any
  other fragment. The refusal is for a metal that survives, i.e. a coordination compound;
* **E/Z double-bond stereochemistry is not assessed** by the stereo step. A library that
  relies on double-bond stereo should be checked with `--stereo refuse` before it is ranked,
  and the absence of a warning is not evidence that the geometry is specified.

---

## 7. The measured cases

Every number below comes from the fixtures the test suite builds, so it can be re-run
offline with no vendor file and no network.

**A sodium salt** (`CC(=O)[O-].[Na+]`): the anion is kept, the counter-ion is reported.

```text
kept — kept the largest organic component CC(=O)[O-] (4 heavy atom(s), charge -1);
removed [Na+] (1 heavy atom(s), charge +1)
```

With `--charge-policy neutralise` the kept molecule becomes `CC(=O)O`, the audit records
`neutralised -1 -> +0 (1 atom(s): O4: -1->+0)`, and the protonation predicate then reports
that the neutral acid is not the species pH 7.4 describes (quoted in §3d).

**A mixture** (`c1ccccc1CCCCCC.O`): the larger organic component is kept and the smaller one
is named.

```text
kept the largest organic component CCCCCCc1ccccc1 (12 heavy atom(s), charge +0);
removed O (1 heavy atom(s), charge +0)
```

**A tautomer pair** (2-hydroxypyridine and 2-pyridone): one molecule survives, and the audit
says which one and why they collided.

```text
smi line 1 hydroxy_pyridine: kept
smi line 2 pyridone: merged into hydroxy_pyridine (tautomer)
```

The two canonical SMILES differ (`Oc1ccccn1` vs `O=c1cccc[nH]1`); their tautomer keys are
equal. With `--exact-dedupe` both survive, which is the control that shows the collapse is
the tautomer key's doing and not a coincidence.

**An unspecified stereocentre** (`CC(F)(Cl)Br`): the policy decides, and the audit says so —
`keep` records `1 centre(s) left unspecified by the input (atoms 2); kept as given`, `drop`
is a no-op because nothing was assigned, and `refuse` rejects the record (quoted in §3e).

**A malformed SMILES**: refused with the line number — `smi line 3 broken: invalid —
'not_a_molecule' is not a parsable SMILES`.

**Property windows**: each violation is listed with its measurement
(`heavy_atoms 3 outside 8..40; molecular_weight 46.07 outside 120..400`).

---

## 8. Why a cleaned library ranks differently — report both

Standardisation changes the answer, on purpose, and the change is not small:

* a **salt** scored as it arrives is scored with its counter-ion, which is charged,
  differently shaped and usually not buried in the pocket;
* a **mixture** scored as it arrives is scored as one of its components chosen by a tool, not
  by a chemist;
* a **neutralised** acid is a different molecule from the carboxylate the pocket sees, and
  the electrostatic term knows it;
* a **tautomer** pair collapses to one molecule, so the ranking loses a row;
* a **dropped** stereocentre merges the enantiomers, so a rank can jump a place for a reason
  that has nothing to do with binding.

So the defensible workflow is:

1. standardise, and **keep the audit**;
2. screen the cleaned library (`odock screen -i clean.smi …`);
3. report both files and the audit next to the results, with the recipe hash, so the reader
   can see which molecules moved and why.

The recipe hash in the header is what makes that reproducible: a downstream `.smi` can be
traced to the exact settings that made it.

---

## 9. What this deliberately leaves to a real registration pipeline

If you need one of these, this tool is the wrong one and it says so:

* **parent-structure extraction across salt forms** — a salt-form database, not a rule;
* **tautomer enumeration and energy ranking** — beyond RDKit's canonical tautomer;
* **stereochemistry inference** from a 2-D depiction — an assignment problem, not a policy;
* **registry identity** — an InChIKey lookup against the database that owns the identifier;
* **legal-name normalisation** of the library's vendor identifiers — nothing here parses a
  vendor catalogue number.

What it *does* give you is a reproducible, auditable answer to "what is in this file, what did
you change, and why" — which is the part a screening campaign actually has to defend.

---

## 10. See also

* [PROTONATION.md](PROTONATION.md) — the charge predicate this recipe asks, and why a charge
  model that cannot hold a formal charge cannot argue about a salt bridge.
* [SCREENING.md](SCREENING.md) — the campaign that consumes the cleaned library.
* [CHEMINFORMATICS.md](CHEMINFORMATICS.md) — the ligand reading and perception layer beneath
  this recipe.
* [`tools/inspect_dist.py`](../tools/inspect_dist.py) — the same "check the artefact, not the
  intention" discipline applied to packaging.

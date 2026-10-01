# Hit triage — structural alerts, drug-likeness and the series view

Ordering a compound, or even spending another docking run on it, needs one more
question answered than the scores do: **what is wrong with this molecule?**
[`odock.triage`](../python/odock/triage.py) answers it with three things that
belong together:

| piece | what it is |
|---|---|
| **structural alerts** | the published catalogues RDKit ships (PAINS, PAINS A/B/C, Brenk, NIH, ZINC) **plus** a documented SMARTS set for the classes a catalogue misses, each reporting *which* alert, *which atoms* and *where* — scaffold, substituent or mixed |
| **property panel** | MW, LogP, TPSA, HBD/HBA, rotatable bonds, aromatic rings, fraction sp³, QED, the SA score when the build has it, judged against **Lipinski, Veber, Egan and Ghose** |
| **the series view** | the whole table grouped by the Murcko series `odock scaffolds` already computes, so "this cluster carries a Michael acceptor, that one is clean" is one screen |

```bash
odock triage -i demo/library.smi
odock triage -i hits.sdf --affinities out/3ptb-screen/results.jsonl --flagged-only
odock triage -i hits.sdf --catalogues PAINS,BRENK,NIH,ZINC -o triage.xlsx --json-out triage.json
```

**What this layer is not.** These are **alerts, not predictions**. An alert is a
literature-derived substructure pattern; a match is not a measurement, and the
false-positive rate measured for an alert set on some other library is not a
probability that *this* molecule will fail. **Absence of an alert is not safety** —
it means nothing in the pattern list matched, which is a statement about the list.
A molecule flagged here can be a good drug; a molecule that passes can fail in the
clinic. Nothing in this file substitutes for an assay, and the CLI prints that
sentence with every report. §4 shows the counts behind it.

## 1. Structural alerts, and where they sit

Two sources, reported side by side:

* **Published catalogues** (RDKit's `FilterCatalog`): PAINS (480 entries), PAINS_A
  (16), PAINS_B (55), PAINS_C (409), Brenk (105), NIH (180), ZINC (50). Each hit
  names the catalogue and the entry (`PAINS:quinone_A(370)`).
* **The embedded SMARTS set**, for the liability classes a catalogue tends to miss
  and for metabolic soft spots. Two families, because "reactive" and "cleared
  quickly" are different decisions:

| family | alerts |
|---|---|
| `liability` | `michael_acceptor_enone`, `michael_acceptor_acrylonitrile`, `michael_acceptor_vinyl_sulfone`, `michael_acceptor_nitroalkene`, `epoxide`, `aziridine`, `acyl_halide`, `sulfonyl_halide`, `alkyl_halide`, `anhydride`, `isocyanate`, `aldehyde`, `nitroaromatic`, `aniline`, `catechol`, `quinone`, `azo`, `hydrazine`, `hydrazone`, `hydroxamic_acid`, `thiol`, `dithiocarbamate`, `n_oxide`, `charged_centre`, `quaternary_nitrogen` |
| `soft_spot` | `para_methoxyphenyl` (O-demethylation), `benzylic_ether` (O-dealkylation), `n_methyl_amide` (N-demethylation), `ester`, `tertiary_amine` |

Every hit carries the matched atom indices and its **location**:

* `scaffold` — every matched atom is part of the Bemis-Murcko framework: the alert
  cannot be removed without changing the chemotype;
* `substituent` — none of them is: an analogue without the alert is one decoration
  away;
* `mixed` — it spans both, which is the case that needs a chemist's eye;
* `none` — the pattern reported no atoms (or the molecule has no atoms to report).

Measured on the bundled molecules, which is what makes the distinction usable:

| molecule | alert | location |
|---|---|---|
| benzoquinone | `PAINS:quinone_A(370)`, 8 atoms | **scaffold** (the quinone *is* the ring system) |
| warfarin | `BRENK:cumarine` | **scaffold** (the coumarin is the fused core) |
| benzamidine | `BRENK:imine_1`, `imine_2` | **substituent** (the amidine hangs off the ring) |
| nitrobenzene | `liability:nitroaromatic`, 4 atoms | **mixed** (nitro group + its ring carbon) |
| caffeine | — | **clean** |

`charged_centre` is deliberately narrow: the charges of a nitro group and of an
N-oxide are reported as *their own* alerts, so a nitro compound is not flagged
twice for the same liability (a test pins that).

## 2. The property panel, and who owns what

Lipinski and Veber come from [`odock.filters`](../python/odock/filters.py) — the
layer the screening pipeline already uses — so a triage table and a screen can
never disagree about a molecule's MW or its PAINS verdict. **`filters.py` owns the
Lipinski/Veber/PAINS verdicts; `triage.py` owns everything else**: the extra
catalogues, the alert locations, and the two rules added here:

| rule | bounds | why it is here |
|---|---|---|
| Lipinski | MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10 | permeability, the classic |
| Veber | RotB ≤ 10, TPSA ≤ 140 Å² | oral bioavailability |
| **Egan** | TPSA ≤ 131.6 Å² **and** LogP ≤ 5.88 | the "drug-like" ellipsoid; it and Lipinski disagree often |
| **Ghose** | MW 160-480, LogP −0.4-5.6, 20-70 heavy atoms, MolMR 40-130 | the ranges the original drug-like set occupied |

The panel also reports TPSA, aromatic ring count, fraction sp³, **QED** and — when
the RDKit build has it — the contrib **SA score**. This build does **not**: the
scorer lives in RDKit's `Contrib` directory, which the binary wheels omit, so
`SA_score` is reported as unavailable (`null` in JSON, an empty CSV cell) with a
note in the report rather than a number of unknown provenance.

A measured caveat that is easy to misread: on the 17-molecule demo library almost
everything fails **Ghose**, because Ghose's lower bound is 160 Da and most of the
demo molecules are 120-160 Da. That is the rule doing exactly what it says, not a
defect in the molecules — a reminder that these rules are descriptions of chemical
space, not verdicts about compounds.

## 3. The triage report

```bash
odock triage -i demo/library.smi
```

```text
triage: 17 molecule(s), 12 with at least one alert, 5 clean

alert rates (a catalogue that flags a large fraction is telling you about the catalogue):
catalogue     flagged    of  fraction
PAINS               1    17       6%
BRENK              11    17      65%
NIH                 2    17      12%
ZINC                1    17       6%

series: 6 group(s) by Murcko scaffold
series                                          n  flagged  alerts
c1ccccc1                                       11        9  ester x1, hydroquinone x1, imine_1 x6, ...
c1ccncc1                                        2        0
O=C1C=CC(=O)C=C1                                1        1  Propenals x1, bis_keto_olefin x1, ...
O=c1[nH]c(=O)c2[nH]cnc2[nH]1                    1        0
O=c1oc2ccccc2cc1Cc1ccccc1                       1        1  cumarine x1
c1ccc2c(c1)c1ccccc1c1ccccc21                    1        1  Polycyclic_aromatic_hydrocarbon_3 x1, ...
```

Three things a chemist reads straight off that:

* **The series view is the point.** The benzene series flags nine of eleven
  molecules while the two pyridines flag none; the coumarin series (warfarin)
  carries a `cumarine` alert **on its scaffold**, so it cannot be edited away,
  whereas the amidine series' `imine_1/imine_2` alerts sit on the **substituent** —
  and the amidine is that series' pharmacophore, which is exactly the tension this
  layer exists to make visible: the catalogue flags the very group that makes the
  series bind, and a chemist has to decide, not the tool.
* **BRENK flags 65 % of this library.** The report prints the rate and says what a
  rate that high means — a statement about the catalogue's breadth, not about the
  molecules. Ranking hits by "number of alerts" on a set like this would be
  mostly ranking BRENK's opinion of benzene rings.
* **Clean molecules stay in the table.** Five of seventeen have no alert at all,
  and they are reported with `clean`, not dropped: a triage table that removed them
  would hide how much of a real library is unremarkable.

Output: a CSV or XLSX with one row per molecule (name, affinity, clean flag,
scaffold, alert count, the alerts with their locations, the failed rules, and the
descriptor panel), plus `--json-out` with the full report including the per-series
summary.

## 4. The counts behind the honesty section

The whole point of the numbers below is that a reader can see the alert rates for
themselves. Measured with every catalogue and both SMARTS sets:

| set | molecules | PAINS | BRENK | NIH | ZINC | SMARTS `liability` | SMARTS `soft_spot` |
|---|---|---|---|---|---|---|---|
| demo library (`demo/library.smi`) | 17 | 1 (6 %) | **11 (65 %)** | 2 (12 %) | 1 (6 %) | 1 (6 %) | 4 (24 %) |
| decoy pool (`demo/decoys.smi`) | 125 | 9 (7 %) | 42 (34 %) | 3 (2 %) | 0 (0 %) | 46 (37 %) | 21 (17 %) |
| both together | 142 | 10 (7 %) | 53 (37 %) | 5 (4 %) | 1 (1 %) | 47 (33 %) | 25 (18 %) |

Read that table before reading any alert-based ranking. On a hand-curated,
fragment-heavy pool the Brenk catalogue flags **a third of the molecules**, and the
SMARTS liability set — deliberately broad, since it reports every aldehyde, aniline,
thiol and charged centre — flags another third. The demo library is flagged harder
still by Brenk (65 %) while its SMARTS liability rate is 6 %, because that set is
six amidines and eleven drug-like molecules: the two sources disagree about which
part of a library is worrying, which is the argument for reporting both. None of
this is a comment on the molecules; it is what these pattern sets do to small,
polar, fragment-rich structures. The report says so in its notes whenever a
catalogue exceeds a quarter of the set, which is why the note is in the output of
§3.

## 5. What this does not establish

* **An alert is a pattern, not a prediction.** It says "this substructure is
  associated with a problem in the literature", not "this molecule will have that
  problem". The association's strength, and the false-positive rate of the
  catalogue, were estimated elsewhere, on other molecules.
* **No alert is not safety.** Absence means nothing matched. A molecule can be
  clean here and fail in an assay, and a molecule can carry five alerts and be a
  marketed drug (many do).
* **The catalogue rates are not probabilities.** §4 gives observed fractions on
  two small curated sets. They are there to be seen, not to be inverted into a
  risk per molecule.
* **Location is about the Murcko framework, not about importance.** "Substituent"
  means the alert is not in the ring/linker system; it does not mean the group is
  unimportant — in the benzamidine series the "substituent" is the pharmacophore.
* **The rules are descriptions of chemical space.** Lipinski/Veber/Egan/Ghose were
  derived from sets of oral drugs; failing one is common among real leads, and
  Ghose fails essentially every fragment-sized molecule.
* **The SA score is often unavailable.** When the RDKit contrib scorer is missing
  the panel reports `None`; no proxy is invented.
* **Nothing here is a synthesis or toxicology assessment.** No route, no cost, no
  metabolite structure, no hERG/AMES prediction — a soft-spot alert is a hint that
  a position *may* be metabolised, not a metabolite.

## 6. Reproducing the numbers

```bash
# the demo library, with the rates, the series view and every alert location
odock triage -i demo/library.smi

# the decoy pool, for the breadth comparison in §4
odock triage -i demo/decoys.smi --top 5

# only the flagged molecules of a screening result, with the affinities
odock triage -i hits.sdf --affinities out/3ptb-screen/results.jsonl --flagged-only \
    -o out/triage.xlsx --json-out out/triage.json
```

```python
from odock import triage
from odock.chem.ligand import read_ligands

hits = read_ligands("demo/library.smi", embed=False)
report = triage.triage_library(hits, affinities={"benzamidine": -5.9})
print(report.rates_table())
print(report.series_table())
print(report.table())
for molecule in report.molecules:
    for alert in molecule.alerts:
        print(molecule.name, alert.label(), alert.atoms)
```

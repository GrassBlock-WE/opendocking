# Projects and reports

A docking run normally ends as a pile of files: a receptor PDBQT, a ligand
PDBQT, a box JSON, a pose PDBQT, an Excel sheet, maybe an SVG diagram. None of
them says which others it belongs with, which settings produced it, or whether
anything has changed since. This page describes what fixes that:

* **a project file** (`*.odockproj`) — one verifiable container holding the whole
  run, which reopens anywhere;
* **an HTML report** (`odock report-html`) — one self-contained document with the
  figures inside it, plus JSON and plain-text siblings for a machine;
* **reproduction and comparison** — re-running a project and diffing runs;
* **a notebook** (`odock project notebook`) — the run as a runnable analysis;
* **studies** — several runs as one named, verifiable, comparable collection
  ([`STUDIES.md`](STUDIES.md)).

```bash
# Wrap a finished run.
odock project save -p poses.pdbqt -r receptor.pdbqt -l ligand.pdbqt \
                   -b box.json -e 16 --seed 42 -o run.odockproj

# Prove it has not changed, and see what it holds.
odock project verify run.odockproj
odock project info   run.odockproj

# Reopen it anywhere: extract the stored files, or re-run the analysis.
odock project open run.odockproj --out reopened/ --reanalyse

# Close the loop: re-run the docking from the stored inputs and compare.
odock project reproduce run.odockproj

# Did a change help? Compare runs side by side, and index a whole campaign.
odock project compare run.odockproj other.odockproj -o comparison.html
odock project index projects/ -o index.html

# Turn it into something a colleague can read.
odock report-html --project run.odockproj -o run.html
```

```python
import odock
from odock import project

box = odock.box_from_ligand(ligand_mol, buffer=8.0)
result = odock.dock("receptor.pdbqt", "ligand.pdbqt", box, exhaustiveness=16, seed=42)

saved = project.save_project(
    "run.odockproj", result,
    receptor="receptor.pdbqt", ligand="ligand.pdbqt", box=box,
    original_inputs={"receptor": "3PTB.pdb", "ligand": "BEN.sdf"},
    engine={"exhaustiveness": 16, "search": "monte_carlo"},
    command=["odock", "dock", "-r", "receptor.pdbqt", "-l", "ligand.pdbqt", "--seed", "42"],
)
print(saved.verify().summary())

reopened = project.open_project("run.odockproj", verify=True)
assert reopened.result().best_affinity == result.best_affinity
```

---

## What a project contains

A `.odockproj` is a ZIP archive. It is written with fixed member timestamps, so
the same content produces the same bytes and two projects can be compared
directly.

| member | contents |
|---|---|
| `project.json` | the manifest: schema and tool versions, the run, the box, the engine settings including the seed, the preparation settings, the input hashes, the analysis, the reproducibility block, and the index of every stored file with its SHA-256 |
| `SHA256SUMS` | `<sha256>  <member>` for every member above, **including `project.json` itself** |
| `files/receptor.pdbqt`, `files/ligand.pdbqt` | the prepared inputs, verbatim |
| `files/inputs/*` | the raw input structures (`--original receptor=3PTB.pdb`) |
| `files/poses.pdbqt` | the poses, as the engine wrote them |
| `files/poses.json` | the same poses with full-precision coordinates, so a reload is exact |
| `files/box.json` | the search box |
| `files/analysis.json` | the ranking table, the best pose's interaction profile, the recurring contacts, the pose-quality numbers and the flags |
| `files/preparation.json` | the preparation reports, when they were supplied |
| `files/figures/*` | any figure stored with the run (a workbench screenshot, an SVG) |

A project is **self-contained**: opening one never reads the original paths. The
manifest may record where a file came from, but only as a normalised,
display-only name (see [Paths](#paths-never-leak) below).

The command line also wraps a screening campaign — one project per docked
molecule, each with its own poses, prepared ligand, receptor and the campaign's
settings:

```bash
odock project screen -s screen_out/ -o projects/ --top 50
```

`odock.screen` itself is not modified by any of this; the campaign's `run.json`
and `results.jsonl` are read as written.

---

## Verification: what it proves, and what it does not

`odock project verify` re-hashes every stored member and compares it with three
things: the manifest, the `SHA256SUMS` member, and the archive's own listing. It
reports, by name:

* files whose bytes changed (the hash and the size);
* members the manifest lists but the archive does not hold;
* members the archive holds but the manifest does not list;
* any disagreement between the manifest and `SHA256SUMS` (which is how a
  hand-edited `project.json` is caught, because its own hash is in the list);
* duplicates inside the archive.

```text
run.odockproj: FAILED (7 of 7 stored file(s) re-hashed, schema 2, 83.0 KiB)
  changed: files/poses.pdbqt: sha256 1ae88352f9e3… -> 2d783d695dc9…
  SHA256SUMS: files/poses.pdbqt: SHA256SUMS says 1ae88352f9e3…, the file hashes to 2d783d695dc9…
```

**A hash detects change; it does not prove correctness.** A matching SHA-256 says
that the bytes are the ones that were stored. It says nothing about whether the
input structure was the right one, whether the preparation did what its recorded
settings claim, whether the force field is appropriate, or whether the pose is
the experimental binding mode. `odock project open` verifies by default for the
same reason: a project whose bytes have changed is not the run it claims to be,
and saying so is cheaper than silently reporting numbers that no longer
correspond to any single run.

---

## The schema version is the compatibility contract

`project.json` carries two version numbers:

* `schema_version` — the layout that was written;
* `min_reader_version` — the oldest reader that can read it.

The reader refuses rather than guesses:

| situation | behaviour |
|---|---|
| **newer** than this build (`schema_version` > 2) | refused, naming both versions, with the advice to upgrade or ask for an export |
| **newer reader requirement** (`min_reader_version` > 2) | refused with the same reasoning |
| **older, with a registered migration** (schema 1) | migrated to the current layout, and the project reports `migrated from 1`; the manifest records what the migration had to infer |
| **older than the migration floor** (schema 0) | refused with the reason, never misread |

A file format that is read "as well as possible" by an older reader produces
plausible-looking wrong poses, which is the failure a version number exists to
prevent. The schema version — not the file extension — is what a reader may rely
on.

---

## The HTML report

`odock report-html` writes one `.html` file per run, with two siblings:

* `<name>.json` — the same numbers, structured, for a script;
* `<name>.txt` — a paste-ready plain-text summary.

The document contains the inputs and their SHA-256, the preparation summary, the
box, the engine settings and the seed, the ranking table, the interaction profile
with the 2-D diagram, the pose-quality metrics — and an explicit **"what this run
does not establish"** section, because the numbers are easy to quote and the
caveats are not.

It is **self-contained** in a checkable sense: figures are inlined (vector SVG in
the document, and a raster preview embedded as `data:image/png;base64`), there is
no stylesheet, script or image fetched from anywhere, and the writer refuses to
return a document that contains a network reference. The raster preview is an
extra: when the headless Qt platform in use has no font database, the preview is
skipped and the report says so, rather than embedding a figure whose labels are
empty boxes.

**A PDF is produced only when a converter is genuinely present** — WeasyPrint, or
a headless Chrome/Edge (`--print-to-pdf`), discovered at run time; neither is a
dependency of this package and nothing is installed on demand. Set
`ODOCK_PDF_CONVERTER` to point at a converter explicitly. When there is none — or
when the converter that exists cannot run (a restricted environment, a browser
that aborts on startup) — the report is written as HTML only and the command says
exactly that. A self-contained HTML with embedded figures is the deliverable; the
PDF is a convenience.

---

## Paths never leak

One release of this project shipped a file that recorded the absolute checkout
path of the machine that produced it, which is worse than useless: it identifies
a stranger's layout and points at a directory that does not exist for them. That
is why path normalisation is built into the container rather than bolted on:

* every string that goes into `project.json` passes through
  `odock.project.portable_path`, recursively, so a forgotten call site cannot
  leak a path;
* a path inside the project's directory or the working directory is recorded
  relative to it, a path inside the user's home becomes `~/…`, and any other
  absolute path is reduced to its **file name**;
* the recorded command has its interpreter path replaced (`python`) and its
  arguments normalised, so it can be pasted into a shell and run;
* the report's JSON block, the HTML and the text sibling are checked for absolute
  paths by the test suite, using the same normaliser.

Structure files themselves are stored **byte for byte**, so a PDBQT whose own
`REMARK` records a local path keeps it — that path is the input's data, not this
tool's metadata. `odock project save` warns when it sees one, naming the file,
so the choice stays visible.

---

## Reproducing a run

A project proves that the *files* have not changed.  `odock project reproduce`
answers the next question: does the run come back?

```bash
odock project reproduce run.odockproj
```

```text
run.odockproj: PASS (tolerance 0 kcal/mol, 0 Å, 2.9 s)
  poses       : stored 2 -> reproduced 2
  best        : stored -6.210 -> reproduced -6.210 kcal/mol (delta 0.000)
  affinity    : max |delta| 0.000 kcal/mol over 2 mode(s)
  top pose    : RMSD 0.000 Å (symmetry-aware, no superposition)
  note        : this is reproducibility, not correctness -- the same code on the
                same inputs; the physics is unchanged.
```

It reopens the project, re-runs the docking **from the stored inputs, box, engine
settings and seed**, and compares pose count, per-mode affinity, the
symmetry-aware RMSD between the two top poses, and the ranking table (including
the residues column).  The outcome is written next to the project as
`<name>.odockproj.reproduce.json` (`--record FILE`, or `--no-record`), and
`--save-as NEW.odockproj` stores the re-run as its own project, carrying the
verdict in its provenance.

### What the verdict means

| verdict | when | what to do |
|---|---|---|
| `PASS` | every compared number is within tolerance | the run reproduces; keep going |
| `FAIL` | it did not match, and every result-deciding setting was recorded | a finding: the code, the recorded settings or the archive is not what it claims |
| `INCONCLUSIVE` | it did not match, but the project never recorded a setting that could explain it (or an input is missing) | record the setting and re-run the reproduction |

`INCONCLUSIVE` exists because of an asymmetry: a `PASS` is a `PASS` whatever was
left unrecorded, but a mismatch cannot be blamed on the engine when the project
never said how many islands, or which search, produced it.  The report names the
missing fields.

### What a tolerance means here

The default tolerance is **zero**, in kcal/mol for the affinity and in Å for the
top-pose RMSD.  That is not optimism: the kernel is deterministic — this
repository's benchmark gate is built on a re-run of a 15-docking campaign
differing in **zero** fields, and `tests/test_project.py` asserts a reproduction
of the 3PTB demo at `max |Δaffinity| = 0.000` and `RMSD = 0.000 Å`.  A non-zero
tolerance is your statement about the noise floor of *another* environment (a
different machine or kernel build can reassociate a parallel floating-point
reduction), and it is recorded next to the verdict so a reader can see which
claim was made.  Choosing a tolerance loose enough to make a mismatch disappear
is choosing not to test.

### What reproduction does not establish

* It re-runs the **same code on the same inputs**.  It does not validate the
  physics, the force field, the preparation or the pose: a bit-identical result
  is evidence about the pipeline, never about the chemistry.
* A `PASS` does not make the affinity experimental, and it does not make the
  ranking meaningful.
* It reproduces the *stored* run.  If the stored run was wrong to begin with, it
  reproduces a wrong run — faithfully.
* It cannot reproduce a run whose inputs the project does not hold.  A project
  saved from a pose file alone, with no receptor or ligand, is reported as
  `INCONCLUSIVE` with the missing pieces named rather than guessed at.
* A changed *stored pose* is caught by `verify` and reported as a `FAIL` with the
  numbers (the re-run disagrees with what the archive now claims).  The
  `files/poses.json` member is the exact record compared against; the
  `files/poses.pdbqt` is the portable artefact, so corrupting *that* is caught by
  `verify` but does not by itself change the comparison — which is why both are
  stored and both are hashed.

---

## Comparing runs

```bash
odock project compare a.odockproj b.odockproj c.odockproj -o comparison.html
```

One self-contained page (plus `.json` and `.txt` siblings) carrying, side by side:

* each run's identity: title, kind, pose count, best affinity, spread, seed,
  force field, ligand efficiency, cluster count and verification state;
* **the engine settings that differ, named field by field** with both values —
  including the case where one side simply did not record a setting, because
  "the settings changed" is not an answer;
* the inputs and their SHA-256, so a change of *experiment* (a different
  receptor, ligand or box) is visible instead of being read as a change of
  settings;
* per-mode affinity deltas against the baseline, and the maximum;
* the top-pose RMSD (symmetry-aware, no superposition) and the Spearman rank
  correlation between the two affinity lists;
* how each pose set clusters at a cutoff, and the difference.

The deltas are arithmetic: the second run minus the first, per mode.  A lower
number is a better score *for this force field in this box*, which is not the
same as a better binder — the page says so, next to the numbers.

## A campaign index

```bash
odock project screen -s screen_out/ -o projects/
odock project index projects/ -o index.html
```

One self-contained page over a directory of projects: the key numbers per member
(poses, best affinity, seed, force field, heavy atoms, ligand efficiency, key
residues, verification state, size), the campaign's docking hash and library
hash, and a link per member — to its HTML report when one sits beside it, and to
the project file itself otherwise.  The rows are ordered by affinity, and the
page states that a row is a summary of a project rather than the run, and that
rows from different campaigns are only comparable when their docking hash
matches.

## A run as a notebook

```bash
odock project notebook run.odockproj -o run.ipynb
```

One `.ipynb` that replays the run: it opens the project file **next to itself**
(the cell looks beside the notebook, then one and two levels up), verifies it,
prints the ranking table, recomputes the interaction profile from the stored
inputs, draws the report's figures inline (embedded as data, so it works
offline), re-runs the docking through `project reproduce`, and ends with the run's
own "what this does not establish" list.  A reader can change a threshold and
re-run one cell.

Two rules shape the document, and both are tested:

* **no absolute path, no URL.**  The notebook names the project by its file name
  and resolves it at run time; the whole document is scanned for drive paths,
  home directories and `http(s)://` references.
* **the claim is labelled.**  `tests/test_notebook.py` executes **every code cell
  in order against a real project, in this interpreter**, stubbing only the
  display helper — that is a real execution check, but it is *not* the claim "it
  runs in a Jupyter kernel".  `odock project notebook --execute` does run it in a
  kernel, through `nbclient`; when `nbclient` or an installed kernel is missing,
  the command writes the notebook anyway and reports plainly that a kernel
  execution **was not verified**.  `docs/STUDIES.md` says the same thing, so
  "notebook export" never implies the stronger claim.

---

## What a project is not

* **A project captures a run, not an interactive session.** The poses, the
  settings and the analysis are frozen. Nothing about the session that produced
  them is recorded: not a box that was dragged by hand before docking, not an
  edit that was undone, not the order in which the inputs were prepared. If a
  session matters, record the commands and the inputs — which is what the
  reproducibility block is for, not a substitute for it.
* **Hashes detect change; they do not prove correctness.** See above.
* **A stored seed reproduces a run only on the same tool version.** The kernel's
  search is deterministic for a given version and seed; a different version may
  legitimately produce different poses. The project records the kernel version so
  that this can be checked rather than assumed.
* **The affinity is a force-field score, not a free energy.** A project makes one
  run reproducible and quotable; it does not make the number experimental.
* **A project is not a database.** It is one run (or one molecule of a campaign).
  Comparing many runs is what `odock screen`, `odock consensus` and
  `odock ensemble` are for; a project is what you archive next to each result.
* **A PDF is not guaranteed.** HTML is the format; a PDF appears when a converter
  happens to be installed.

---

## Tests

`tests/test_project.py` and `tests/test_htmlreport.py` cover the promises this
page makes, and the ones that matter are deliberately awkward:

* a completed docking run is saved, then reopened **in a fresh interpreter, in
  another directory, with the original paths gone**, and the reloaded poses, box,
  settings and a recomputed analysis are compared field for field with the
  original;
* the run is **reproduced from its own project** — `max |Δaffinity| = 0.000`,
  top-pose `RMSD = 0.000 Å` — and reproducing it twice gives the same evidence;
* one recorded setting is changed and `reproduce` reports a `FAIL` with the
  numbers instead of passing; a project that never recorded a deciding setting
  yields `INCONCLUSIVE` with the fields named; a truncated stored pose record is
  a reported failure rather than an exception;
* two genuinely different runs compare with named setting differences, real
  per-mode deltas and a top-pose RMSD;
* one byte of one stored file is flipped and `verify` has to name that file; a
  missing member, an unlisted member and a hand-edited manifest are caught too;
* a newer schema is refused with both version numbers in the message, an older
  one is migrated, and one below the floor is refused with the reason;
* the HTML contains no `http://` or `https://` reference at all, and every
  embedded PNG is inflated and checked to be a complete image;
* a hostile title (`<script>…</script>`) is escaped and cannot inject an element;
* a project and a report built from absolute paths contain no absolute path.

```bash
python -m pytest tests/test_project.py tests/test_htmlreport.py -q
```

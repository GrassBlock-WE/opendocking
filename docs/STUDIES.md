# Studies

A **project** is one run.  A **study** is a named collection of runs: the thing
you compare when you ask "did the protocol change help?", and the thing you hand
over when you say "here is the series".

```bash
# A study with its shared metadata and its protocol.
odock study create screen-2026-01 \
    --name trypsin-series \
    --title "3PTB benzamidine series" \
    --target "bovine trypsin" \
    --receptor receptor.pdbqt \
    --library library.smi \
    --protocol scoring=vina --protocol exhaustiveness=16 --protocol seed=42 \
    --operator alice \
    --project out/benzamidine.odockproj \
    --project out/benzamidine_methyl.odockproj

# Add more runs later, verify the whole collection, compare two versions.
odock study add screen-2026-01 --label benzylamidine --project out/benzyl.odockproj
odock study verify screen-2026-01
odock study diff screen-jan screen-feb --top 10 -o diff.html
odock study report screen-2026-01 -o study.html
```

```python
from odock import study

created = study.create_study(
    "screen-2026-01",
    name="trypsin-series",
    target="bovine trypsin",
    library="library.smi",
    protocol={"scoring": "vina", "exhaustiveness": 16, "seed": 42},
    operator="alice",
    members=["out/benzamidine.odockproj"],
)
print(created.verify().summary())
```

---

## What a study is

```text
screen-2026-01/
    study.json          format, schema version, name, metadata, protocol,
                        audit block, and every member with its SHA-256
    SHA256SUMS          <sha256>  <member> for study.json and every member
    members/*.odockproj the member projects (copied in)
    *.html              optional: a member's report, beside it
```

* **Shared metadata**: `target`, `receptors`, `library`, a free-form `protocol`
  mapping, `notes`, plus any other field you record.  The protocol is what a diff
  compares, so it is worth filling in: `--protocol scoring=vina
  --protocol exhaustiveness=16` stores a string and a number respectively.
* **Per-member hashes**: the manifest records the SHA-256 of every member project
  as it was when the member was added.  `odock study verify` re-hashes the files
  on disk and reports a changed or missing member **by name**; it also verifies
  each member project's *own* internal hashes, so a study is checked at both
  levels, and it checks `SHA256SUMS` (which covers `study.json` itself, so a
  hand-edited manifest is caught too).
* **Self-contained by default**: members are **copied** into `members/`.  With
  `--link` the manifest records a path relative to the study directory instead
  (a project tree already on disk stays where it is) — the study then verifies
  only while that relative path resolves.  Either way, no absolute path is ever
  written: every string in `study.json` passes through
  `odock.project.portable_path`.
* **Audit**: `study.json` records the operator label, the tool and kernel
  versions, the Python version and the platform, with the creation time — so a
  directory of studies can be attributed without guessing from file ownership.
* **Versioned like a project**: `schema_version` (and `min_reader_version`) are in
  the manifest, and a study written by a newer OpenDocking is refused with both
  version numbers named rather than read approximately.

---

## Diffing two studies

`odock study diff A B` answers three questions with numbers rather than prose:

1. **What changed in the protocol?**  Every metadata and protocol field that
   differs, named, with both values, and a note when one side never recorded it:

   ```text
   Protocol and metadata that changed
   ----------------------------------
     library: library.smi -> library-v2.smi
     protocol.exhaustiveness: 16 -> 32
     protocol.seed: 42 -> 7
   ```

2. **What changed in the collection?**  Members shared, added and removed.

3. **How did the hit list move?**  For every member the two studies share:

   | column | meaning |
   |---|---|
   | `best A`, `best B` | the best affinity recorded in each study (kcal/mol) |
   | `Δbest` | B minus A: positive is worse |
   | `rank A`, `rank B` | the member's rank **within its own study** (1 = best) |
   | `Δrank` | B minus A: positive means it slipped down its own pool |
   | `movement` | `entered the top N`, `left the top N`, `stayed in the top N`, or, for a member outside the top N on both sides, `improved` / `worsened` / `unchanged` |

   The N is stated on the page and in the JSON (`top_n`), and the top-N sets on
   each side are listed (`top_a`, `top_b`, `entered_top`, `left_top`).  Measured
   example from this tree:

   ```text
   How the hit list moved (top 1)
   --------------------------------
     member          best A   best B    dBest  rank A  rank B  dRank  movement
     benzamidine     -6.210   -5.903    0.307       1       1     +0  stayed in the top N
     benzamidine_methyl -5.474 -5.474   0.000       2       2     +0  unchanged
     entered the top 1: none
     left the top 1: none
   ```

   Each shared member's pairwise numbers (Δbest, the largest per-mode difference,
   the symmetry-aware top-pose RMSD and the Spearman rank correlation) come from
   `odock.project.compare_projects` applied to that member's two projects, so
   there is exactly one implementation of those deltas in the tool.

The diff page is one self-contained HTML file (`-o diff.html`) built on the same
document builder as the comparison and single-run reports, with `.json` and
`.txt` siblings; it contains no `http`/`https` reference and no absolute path.

---

## The study report

`odock study report DIR -o study.html` writes one self-contained page: the
protocol and metadata, every member with its verification state and key numbers
(poses, best affinity, seed, force field, heavy atoms, ligand efficiency, key
residues, size), the top-N table across the study, and a link per member to its
report when one sits beside it (`<label>.html`, `<label>.html` in `members/`, or
a name given in the report payload).

Member links are relative, and when the page is written **outside** the study
directory the writer rewrites them relative to the page, so a page copied
somewhere else still resolves its links.

---

## What a study does not establish

* A study is a **collection, not a control**: it says which runs were grouped and
  with which protocol, not that the grouping is a fair experiment.
* The member hashes show that nothing has changed since a member was added; they
  do not show that a run is correct.
* **A rank delta is a movement within its own study.**  Two studies with
  different members have different pools, so a rank can move because the pool
  changed rather than because the molecule did.
* A member that enters a top-N scored better *relative to that pool*, which is
  not the same as being a better binder.
* Comparing studies run with different force fields compares two scales.
* Nothing here is a statistical test: the deltas are arithmetic on the recorded
  numbers, with no error model behind them.

## Related

* [`PROJECTS.md`](PROJECTS.md) — the single run: the container, verification, the
  schema contract, `reproduce`, `compare`, the campaign index, and the
  **notebook export**.  Note what a notebook export claims: its code cells are
  executed against a real project *in-process* in the test suite, while an actual
  Jupyter-kernel execution is reported as **not verified** unless `nbclient` and
  a kernel are installed.
* `odock report-html` — the report of one run, which a study report links to.

## Tests

`tests/test_study.py` covers the promises above: create/add/verify (including a
member changed byte-by-byte, reported by name), a study with its own metadata and
audit block, a linked member recorded as a relative path, a newer schema refused
with both versions, the diff's protocol fields and membership changes, the
hand-checkable rank deltas and top-N movement, a study diffed against itself
showing nothing, and both pages asserted self-contained (no URL, no absolute
path, every link resolvable).

```bash
python -m pytest tests/test_study.py tests/test_notebook.py -q
```

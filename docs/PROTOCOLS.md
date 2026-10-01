# Protocols

*A protocol is every decision a run depends on, as one versioned, diffable
document.*

A working pipeline used to live in a shell history: `prepare` flags, the box and
how it was derived, engine and search settings, the seed, the filters and their
thresholds were typed once, worked, and were never recorded. A lab that found
settings that work could not hand them to a collaborator, and could not tell
whether a colleague's run used the same protocol.

## The limit, first

A protocol captures **settings, not intent or provenance**:

* it cannot tell you that the receptor was the right one — that the file is the
  right protein, the right chain, the right protonation state, or that the box
  covers the site you care about;
* it does not record *why* a setting was chosen, only that it was;
* its **hash proves identity, not correctness**: two runs whose recorded hash
  matches were asked for the same things. That is a precondition for comparing
  their numbers, and nothing more. A protocol can be followed exactly and still
  produce a meaningless result.

Everything below is about making the *settings* shareable. Read this section
again before you cite a hash in a paper.

## What is in a protocol

| section | what it records |
|---|---|
| `preparation` | `keep_water`, `keep_hetero` (the CLI's `--no-hetero` inverts it), `strip`, `add_polar_hydrogens`, `strict`, the `modified_residues` set the `--no-hetero` semantics keep, the ligand preparation flags (`add_hydrogens`, `strip_nonpolar`, `rigid_amides`, `embed`, `optimize`) and `prepare_seed` |
| `box` | `source` (`ligand`, `pocket`, `manual`, `receptor` — **how** the box was derived), `center`, `size`, `spacing`, `ligand_padding`, `pocket_index` |
| `engine` | `scoring`, `search`, `exhaustiveness`, `num_poses`, `min_rmsd`, `energy_range`, `islands`, `population`, `generations`, `use_grid`, `refine` |
| `execution` | `seed`, `jobs`, `timeout`, `checkpoint_every` |
| `library` | `filters`, `limit`, `top`, `format`, `interactions`, `write_poses`, `consensus`, `consensus_top`, `consensus_method` |
| `thresholds` | `interactions` (H-bond, salt bridge, hydrophobic, π, cation–π, clash ratio), plus free-form `pocket` and `strain` blocks |
| `provenance` | the tool, kernel and Python versions that wrote the document — **excluded from the hash** |

The box's `source` is recorded because "the same numbers" reached by different
routes are different decisions: a box fitted to the ligand follows it to a new
pose, a hand-placed one does not.

## Commands

```console
$ odock protocol list                          # the shipped templates, with their note
$ odock protocol show   --template fast-screen # every setting, and the hash note
$ odock protocol validate mine.json            # version, fields, portability, run-readiness
$ odock protocol diff   mine.json yours.json   # field by field, both values
$ odock protocol hash   mine.json              # the identity, and what it covers
$ odock protocol save   mine.json --template fast-screen \
      --set engine.exhaustiveness=16 --name "our screen"
$ odock protocol run    mine.json -r receptor.pdbqt -i library.smi -o out/run
$ odock protocol run    mine.json -r r.pdbqt -i l.smi -o out/run --dry-run
```

`diff` exits **1** when the two protocols differ, which makes it usable in a
script: "does this run use our protocol?" is one command.

`--dry-run` assembles and validates everything — including the box — and stops
before docking, so a protocol can be checked without spending an afternoon.

## The shipped templates

Each is a reviewable JSON file in `examples/`, and `odock protocol list` prints
the note:

| template | use it when |
|---|---|
| `fast-screen` | first pass over a whole library: rank molecules, do not quote poses |
| `careful-redock` | redock a known ligand into its own structure, or defend a number in a paper |
| `ensemble` | one crystal structure is not the whole story (a flexible pocket, several snapshots) |
| `fragment-pass` | fragments and very small ligands, where the usual scoring is noisy |

The templates are also **validation input**: the test-suite loads every one of
them, requires each to be runnable, and requires them to stay distinct — a
template that rots fails a test rather than a user.

> **Packaging note.** The templates live in `examples/` because that directory is
> already listed in the sdist `include` list in `pyproject.toml`. Adding a new
> top-level directory would need that file changed by its owner.

## What the hash covers, and what it does not

```
hash = sha256(canonical JSON of {schema_version, preparation, box, engine,
                                 execution, library, thresholds})
```

Canonical means sorted keys, no whitespace, ASCII only — so two labs on two
machines that recorded the same decisions compute the same digest.

Deliberately **not** in the hash:

| excluded | why |
|---|---|
| `name`, `note` | labels, not decisions. Two templates with identical settings hash alike, which is correct |
| `provenance` | tool/kernel/Python versions and the creation time describe the machine and the moment, not the decision |
| any path | a path is a location, not a decision. The **inputs'** identity is recorded separately, by the run's own input digests (`library_hash`/`docking_hash` in `screen.py`) |

So the same settings on two machines give the same hash, while the fact that they
were run on *different receptors* does not change it — which is why the run
records both the protocol hash **and** its input digests.

## Reproducibility: what a run records

`odock protocol run` writes into the output directory:

* `protocol.json` — the document that was run, so it can be re-run or diffed
  without recovering it from the original file;
* `protocol-hash.txt` — one line: the digest, the name and the schema version;
* the run's own `manifest.json` gains a `protocol` block (`name`,
  `schema_version`, `hash`, `hash_note`, `document`).

A third party can then ask the question both ways:

```console
$ odock protocol hash out/run/protocol.json       # recompute it
$ odock protocol diff out/run/protocol.json mine.json   # compare it
```

or, from the API (and from the console dock):

```python
>>> from odock import protocol
>>> protocol.verify_run("out/run", protocol.load_protocol("mine.json"))
(True, 'the run records protocol 9d15b6b8…, which is exactly this protocol (mine, schema 1)')
```

That answer is *identity*, not quality: read the limit section again before
quoting it.

## In the workbench

`View ▸ Panels ▸ Inspector ▸ Protocol` shows the settings the session would run
with, rendered as a protocol — including the hash note — with:

* **Save protocol…** — write the current settings as a validated document;
* **Load protocol…** — put a saved document back into the controls it describes
  (box, scoring, search, exhaustiveness, poses, energy window, islands,
  population, generations, seed, threads, interaction thresholds);
* **Compare with saved** — a field-by-field diff in the log;
* **Template** — "what differs from this template", computed *before* you run
  something expensive, with the full diff in the tooltip.

The tab holds no settings of its own: it reads the widgets, so a document can
never describe something the workbench is not set to do. What the GUI has no
control for (hetero/water handling, preparation flags, library filters) stays at
the documented default in a captured document rather than being invented; use
`protocol save --set section.field=value` for those.

The same functions are bound in the console, so the keyboard-only path reaches
them too:

```python
>>> current_protocol()                    # the live settings
>>> save_protocol(current_protocol(), "mine.json")
>>> diff_protocols(load_protocol("mine.json"), current_protocol())
>>> run_protocol(load_protocol("theirs.json"), receptor="r.pdbqt",
...              library="l.smi", outdir="out/theirs")
```

## Validation, and why it is strict

| refusal | message names |
|---|---|
| unknown schema version | both versions, and which side to upgrade |
| unknown field | the field and its path (`protocol.engine.exhaustivness`) plus the fields that do exist |
| wrong type | the field and what was expected |
| missing required field | the field — **before** a run starts, not after the first molecule |
| absolute path | the field and the value; a protocol must be portable between machines |

The strictness is the feature. A protocol that silently drops a setting still
runs and still produces numbers — attributed to settings that were not used,
which is the failure mode this file exists to prevent.

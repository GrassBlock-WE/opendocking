# The documentation site

```bash
odock docs build              # render into out/docs-site (ignored by git)
odock docs build -o /tmp/site # anywhere else
odock docs check              # build and guard it; non-zero on an error
odock docs check --json       # the same, machine-readable (this is the release gate)
odock docs serve              # optional: serve the built site on 127.0.0.1
```

Two dozen documents live in `docs/`, `demo/` and the repository root, and the only
way to read them used to be one file at a time on GitHub with no index and no
search. This renders them into one directory: an index grouped by audience, a page
per document with a sidebar, working anchors, resolving cross-links, a search
index, and a guard that fails on the defects a reader would hit.

---

## What gets built

* **A page per document**, plus an `index.html`.  Page names are flat and
  collision-free (`docs/USER_GUIDE.md` → `docs_user_guide.html`,
  `demo/README.md` → `demo_readme.html`), so every link between pages is a bare
  file name and no page needs to know where it sits.
* **An index grouped by audience**: getting started, the science, the workbench,
  engineering, release.  The grouping is a table in `python/odock/docs.py`
  (`DOC_AUDIENCE`).  A document nobody filed still appears — under "More
  documents" — and the build **reports** it, so nothing is ever dropped and nothing
  is silently uncategorised.
* **The source set is discovered, not listed.**  It is every *published* markdown
  document under `docs/`, the repository root and `demo/`, decided by `.gitignore`
  through the same engine `release stage` uses.  A new `docs/*.md` appears in the
  site and the sidebar on the next build without anybody remembering to add it; and
  the guard fails if the generated set and the source set ever disagree.
  Two consequences worth knowing: an internal document that `.gitignore` excludes
  (the requirements brief, the hand-over) is **not** rendered — the site cannot leak
  what a release refuses to ship — and `.github/` is excluded on purpose, because an
  issue template is a form, not documentation.
* **One visual language.**  The stylesheet is
  `odock.htmlreport.DOCUMENT_CSS` — the same one the HTML reports use — plus the
  layout rules a multi-page site needs (sidebar, search box, code blocks).  There is
  no second look to keep in sync.
* **The markdown subset these documents use**: ATX headings, paragraphs, fenced
  code with an optional language, pipe tables with alignment, bullet and numbered
  lists nested by indentation, blockquotes, horizontal rules, and inline code,
  bold, italic, links, images and autolinks.  **Raw HTML is escaped, never passed
  through.**  The documents contain none of it, and escaping means a document
  cannot inject a resource the guard would have to catch.  Indented code blocks and
  reference-style link definitions are *not* interpreted; they render as text
  (visible, not broken).
* **Anchors match GitHub's.**  Heading ids are `github-slugger` ids (lowercase,
  punctuation dropped, spaces to hyphens, repeated hyphens **kept**, letters outside
  ASCII kept), so the `#fragment` links the documents already contain resolve here
  as they do there.  The guard additionally accepts the collapsed spelling of an
  anchor, so a link written either way resolves.

## The search index

Built at generation time, one entry per section: the page, the document title, the
heading, its anchor, the audience group, and a 200-character extract of that
section's own prose (code fences excluded).  It is written as
`assets/search.js` — a **script**, not a fetched JSON file — because a `file://`
page cannot `fetch()` one, and the site must work from disk.  Matching is
case-insensitive **substring** over the title, heading and extract.

What it can and cannot do, plainly:

* it **can** find a word or phrase in a heading or in the first 200 characters of a
  section, and it lands on the heading that contains it;
* it **cannot** stem (`docking` does not match `dock`), rank by relevance (it shows
  the first matches in page order, capped at 12), search inside code fences, or find
  a word that appears only deep inside a long section.
* the size is reported by every build: currently about **150 KB** for 450-odd
  sections over 24 documents (a fifth of that gzipped).  The extracts dominate it;
  dropping them would cut it to about 60 KB at the cost of body search.

## The guard: `odock docs check`

| finding | severity | what it means |
|---|---|---|
| `docs.links` | **error** | a cross-document link or an in-page anchor does not resolve. The detail names the source document and the target |
| `docs.network` | **error** | a page references something the browser would **fetch** remotely (a remote `src`, `<link>`, `url()`, `@import`, a frame). External **hyperlinks** are allowed — a citation, or a link to a source file — and are counted and listed |
| `docs.images` | warning | a figure a document references is not in this tree. Generated figures (`out/…`) are a warning by design, since a clone does not have them; `--strict-images` makes them errors |
| `docs.titles` | **error** | a page without an `<h1>` (the build supplies one from the file name, so this is a bug in the build, not in a document) |
| `docs.set` | **error** | the generated set differs from the source set — a document that would silently not be published |
| `docs.search` | ok | the index size and entry count, always reported so a jump is visible |

The guard distinguishes a **resource** from a **link** on purpose.
`find_external_references` in `htmlreport` exists for reports built from run data,
where an absolute URL in *any* attribute is a leak; here the question is narrower —
does the page fetch anything? — because a documentation site legitimately cites
things.  So `<a href="https://keepachangelog.com/…">` is a link (counted), while
`<script src="https://cdn…">` is an error.

**What it does not check: whether a statement in a document is true.**  It checks
structure and links: that a page exists for every document, that a link goes
somewhere, that a figure is there, that a title is present, and that nothing is
fetched.  A document can be perfectly linked and completely wrong.

## Serving it

`odock docs serve` runs a read-only HTTP server on `127.0.0.1` for convenience.
It is deliberately **not** the way a release is read: the pages carry their own
stylesheet and load the search index from the same directory, so opening
`index.html` from the filesystem works.  The server exists because a local address
is easier to read on a tablet, and the release gates never use it.

## In a release

`release check` runs `odock docs check` as a gate: a broken documentation link
fails a release instead of shipping.  The gate builds into `out/docs-site`
(ignored by git) so that running the gate cannot add a directory to the published
set — a build product in the repository root would otherwise be content that
`release stage` would happily package.

## The API reference

One page per module under `python/odock/` (including `odock.chem` and `odock.gui`),
grouped in the sidebar under **API reference**, built from the **source**: the
module's public surface — the names its `__all__` declares, or every non-underscore
top-level class and function when it declares none — with each signature, each
docstring, and each public method.

It is read with `ast`, **never by importing**. That is a safety property, not a
performance one: importing 59 modules to describe them would execute module-level
code (a Qt import, an environment read, a file) and a documentation build must not
run the thing it documents. It also means the reference works for a module that
cannot be imported here at all — `tests/test_docs_site.py` pins that with a module
that raises at import time and is described anyway.

A public name with **no docstring is reported, not omitted**: it appears on its page
with a line saying so, and the guard's `docs.reference` finding carries the counts.
On this tree: **59 modules, 1 628 public names, 550 without a docstring** — and the
composition matters more than the number: they are dominated by `@property`
accessors and `as_dict`/`to_dict` serialisers, which are public API but rarely
documented individually. The number is reported rather than smoothed away.

### What the reference does not cover

* **Signatures, not semantics.** It shows what the arguments are called and what the
  docstring says. It cannot tell you whether the implementation does that, what it
  costs, or when it raises — a beautifully documented function can be wrong.
* **No examples are run.** An example inside a docstring is rendered as text and
  never executed, so one that has drifted stays wrong quietly. The tutorial exists
  for the part that must be executed.
* **Only the shape the source declares.** A name exported dynamically, a
  `getattr`-based façade, a type created at run time: none appear. If a name is not
  written in the module body, it is not in the reference.
* **Not a type checker.** Annotations are quoted from the source, not resolved, so
  `Optional[PathLike]` is shown as written rather than as its expansion.

## The tutorial

`odock tutorial` walks the whole toolchain on the bundled 3PTB data — prepare the
receptor, build the ligand from `demo/library.smi`, choose the box, dock, read the
ranking, profile the interactions, save a project and verify it reproduces, export
the report — and prints what each step **measured**. It takes about five seconds,
writes only under `out/`, is seeded, and asserts the shape of every step, so it
fails loudly instead of printing a plausible transcript.

The site's tutorial page is rendered from a **run**: `docs build` executes the
tutorial and renders the record, so the numbers on the page are the numbers of that
build and the page cannot drift from the code. `--json` prints the same record, and
`tests/test_docs_site.py` compares a page built here with a tutorial run in a
**fresh interpreter** — a number measured in the subprocess must appear on the page.
A tree without the bundled demo data skips it with a note rather than pretending.

### What the tutorial does not cover

* **It proves the happy path, not correctness.** Every step asserts that a result
  came back with the shape it should have. A wrong score, a bad pose or a
  scientifically invalid box all pass happily: catching those is
  `docs/VALIDATION.md` and the benchmark's job, not a tutorial's.
* **It runs on one small system.** 3PTB, one ligand, exhaustiveness 2 — chosen to be
  fast. Nothing about a large receptor, a flexible side chain, a metal, or a
  screening campaign follows from it.
* **It is not a benchmark.** The numbers are for orientation, not comparison: one
  seeded run, and a different machine or a rebuilt kernel moves them.
* **It does not cover the GUI.** The workbench has its own entry point; the tutorial
  deliberately stays on the command-line API.

## Tests

`tests/test_docs_site.py`: the renderer on each construct (including that raw HTML
is escaped and that inline code survives the other inline rules), the source-set
discovery (including that a new document appears without being listed and that an
unfiled one is reported not dropped), the site (pages, index, sidebar, rewritten
cross-links, copied images), the guard firing on each defect in a simulated tree,
the search index's shape and its per-section extracts, **a real HTTP fetch** from
`serve_site` (started, fetched, shut down — not a mock), the API reference
discovering a module added to a temporary package and reporting an undocumented
public name, the reference reading a module that raises at import time, the tutorial
page rendered from a run, the tutorial run in a **fresh interpreter** with its
measured number found on the page, and the repository's own documentation set, which
is the gate's real input.

```bash
python -m pytest tests/test_docs_site.py -q
```

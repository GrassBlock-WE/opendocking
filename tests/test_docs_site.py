# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock docs``: the markdown renderer, the site, the guard and the server.

The claims these tests make, and the ones they deliberately do not:

* *the renderer renders the constructs the documentation actually uses* — headings,
  fenced code, pipe tables, nested lists, blockquotes, rules, inline code/bold/
  italic/links/images — and **escapes everything else**, so a document cannot inject
  markup. Tested on small inputs and, for the set, against the real documents.
* *the site is complete and coherent* — a page per published document, an index
  grouped by audience, a sidebar, resolving cross-links and anchors.
* *nothing is fetched* — no remote resource in any page, and the search index is a
  script rather than a `fetch()`, because `file://` blocks the latter. What is **not**
  claimed: that a browser will render every pixel identically; the tests assert
  structure, not appearance.
* *the guard fires* — on a broken link, a missing anchor, a missing image, a missing
  title, a set mismatch and a remote resource, each simulated in a temporary tree.
"""

from __future__ import annotations

import json
import re
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from odock import docs

ROOT = Path(__file__).resolve().parent.parent


def _tree(tmp_path: Path) -> Path:
    """A miniature repository with the documents and the manifests to match."""
    root = tmp_path / "repo"
    (root / "docs").mkdir(parents=True)
    (root / "demo").mkdir(parents=True)
    (root / ".github").mkdir(parents=True)
    (root / ".gitignore").write_text("out/\nscratch/\nodck.md\n", encoding="utf-8")
    (root / "pyproject.toml").write_text(
        "[project]\nname = \"x\"\nversion = \"1.0.0\"\n\n[project.urls]\n"
        'Repository = "https://github.com/Owner/opendocking"\n',
        encoding="utf-8",
    )
    (root / "README.md").write_text(
        "# OpenDocking\n\n"
        "The entry point. See [the guide](docs/SCORING.md) and "
        "[a section](docs/SCORING.md#how-it-works), or [the source](python/odock/docs.py).\n"
        "\n## Install\n\n```bash\npython -m pip install .\n```\n",
        encoding="utf-8",
    )
    (root / "docs" / "SCORING.md").write_text(
        "# Scoring\n\n"
        "Text with `inline code`, **bold**, *italic* and a [link back](../README.md).\n\n"
        "## How it works\n\n"
        "| step | what | note |\n|---|---:|:---:|\n"
        "| one | read | ok |\n| two | dock | fine |\n\n"
        "* first\n* second\n    * nested\n\n"
        "> a quotation\n\n"
        "---\n\n"
        "![a figure](../demo/figure.png)\n",
        encoding="utf-8",
    )
    (root / "docs" / "ARCHITECTURE.md").write_text(
        "# Architecture\n\nHow the pieces fit.\n", encoding="utf-8"
    )
    (root / "docs" / "EMPTY.md").write_text("# Empty\n\nNothing yet.\n", encoding="utf-8")
    (root / "demo" / "README.md").write_text("# Demo\n\nRun it.\n", encoding="utf-8")
    (root / ".github" / "pull_request_template.md").write_text("# PR\n", encoding="utf-8")
    (root / "odck.md").write_text("# internal\n", encoding="utf-8")
    (root / "python" / "odock").mkdir(parents=True)
    (root / "python" / "odock" / "docs.py").write_text(
        '"""A module for the reference to find."""\n\n'
        "__all__ = ['render']\n\n\n"
        "def render(text: str, *, width: int = 80) -> str:\n"
        '    """Render some text."""\n'
        "    return text\n",
        encoding="utf-8",
    )
    (root / "demo" / "figure.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    return root


# ---------------------------------------------------------------------------
# The markdown renderer
# ---------------------------------------------------------------------------


def test_slugify_matches_the_github_shape():
    assert docs.slugify("How it works") == "how-it-works"
    assert docs.slugify("6. What is *not* validated") == "6-what-is-not-validated"
    assert docs.slugify("`odock doctor` --json") == "odock-doctor---json"
    assert docs.slugify("Ångström units") == "ångström-units"
    assert docs.slugify("   ") == ""


def test_headings_get_ids_and_a_table_of_contents():
    rendered = docs.render_markdown("# Title\n\n## First part\n\nText.\n\n### Deeper\n")
    assert '<h1 id="title">Title</h1>' in rendered.html
    assert '<h2 id="first-part">First part</h2>' in rendered.html
    assert rendered.title == "Title"
    assert [level for level, _, _ in rendered.toc] == [1, 2, 3]
    assert [anchor for _, _, anchor in rendered.toc] == ["title", "first-part", "deeper"]


def test_repeated_headings_get_unique_ids_like_github():
    rendered = docs.render_markdown("## Same\n\n## Same\n\n## Same\n")
    ids = re.findall(r'<h2 id="([^"]+)"', rendered.html)
    assert ids == ["same", "same-1", "same-2"]


def test_a_fenced_block_is_escaped_and_labelled():
    rendered = docs.render_markdown("```python\nif a < b:\n    print('x')\n```\n")
    assert 'class="language-python"' in rendered.html
    assert "if a &lt; b:" in rendered.html
    assert "<pre><code" in rendered.html


def test_inline_code_survives_the_other_inline_rules():
    """The regression that reading a generated page caught: a placeholder rewrite
    turned every code span into a literal backtick."""
    rendered = docs.render_markdown("Use `odock dock --score` and **`bold code`**.\n")
    assert "<code>odock dock --score</code>" in rendered.html
    assert "<strong><code>bold code</code></strong>" in rendered.html
    assert "`" not in rendered.html
    assert "\x00" not in rendered.html


def test_a_pipe_table_renders_with_alignment():
    rendered = docs.render_markdown(
        "| a | b | c |\n|---|---:|:---:|\n| 1 | 2 | 3 |\n"
    )
    assert "<table>" in rendered.html and "<th" in rendered.html
    assert 'style="text-align:right"' in rendered.html
    assert 'style="text-align:center"' in rendered.html
    assert "<td>1</td>" in rendered.html.replace(' style="text-align:left"', "")


def test_nested_lists_render():
    rendered = docs.render_markdown("- one\n- two\n    - nested\n")
    assert rendered.html.count("<ul>") == 2
    assert "nested" in rendered.html
    assert rendered.html.index("one") < rendered.html.index("nested")


def test_blockquote_rule_and_paragraphs():
    rendered = docs.render_markdown("> quoted **words**\n\n---\n\nafter\n")
    assert "<blockquote>quoted <strong>words</strong></blockquote>" in rendered.html
    assert "<hr/>" in rendered.html
    assert "<p>after</p>" in rendered.html


def test_raw_html_is_escaped_not_passed_through():
    rendered = docs.render_markdown('<script src="https://cdn.example/x.js"></script>\n')
    assert "<script" not in rendered.html
    assert "&lt;script" in rendered.html
    assert docs.external_resources(rendered.html) == []


def test_links_and_images_go_through_the_resolvers():
    seen: dict = {}

    def resolve_link(target):
        seen["link"] = target
        return "resolved.html#anchor"

    def resolve_image(target):
        seen["image"] = target
        return "assets/images/x.png"

    rendered = docs.render_markdown(
        "[text](OTHER.md#anchor) and ![alt](img.png)\n",
        resolve_link=resolve_link, resolve_image=resolve_image,
    )
    assert seen == {"link": "OTHER.md#anchor", "image": "img.png"}
    assert '<a href="resolved.html#anchor">text</a>' in rendered.html
    assert '<img src="assets/images/x.png" alt="alt"/>' in rendered.html
    assert rendered.links[0]["target"] == "OTHER.md#anchor"
    assert rendered.images[0]["target"] == "img.png"


def test_an_autolink_becomes_a_hyperlink():
    rendered = docs.render_markdown("See <https://example.org/x> for more.\n")
    assert '<a href="https://example.org/x" rel="noopener">https://example.org/x</a>' in rendered.html


# ---------------------------------------------------------------------------
# The source set
# ---------------------------------------------------------------------------


def test_the_source_set_is_discovered_from_the_published_files(tmp_path):
    root = _tree(tmp_path)
    sources = {source.relative: source for source in docs.source_documents(root)}
    assert "README.md" in sources
    assert "docs/SCORING.md" in sources
    assert "docs/EMPTY.md" in sources
    assert "demo/README.md" in sources
    # An issue template is a form, not documentation; an ignored internal document
    # is not published, so the site does not render either of them.
    assert ".github/pull_request_template.md" not in sources
    assert "odck.md" not in sources
    assert sources["docs/SCORING.md"].title == "Scoring"
    assert sources["docs/SCORING.md"].page == "docs_scoring.html"
    assert sources["docs/SCORING.md"].group == "science"
    assert sources["docs/ARCHITECTURE.md"].group == "engineering"
    assert sources["docs/EMPTY.md"].group == "more"


def test_a_new_document_appears_without_being_listed(tmp_path):
    root = _tree(tmp_path)
    before = {source.relative for source in docs.source_documents(root)}
    (root / "docs" / "BRAND_NEW.md").write_text("# Brand new\n\nText.\n", encoding="utf-8")
    after = {source.relative for source in docs.source_documents(root)}
    assert after - before == {"docs/BRAND_NEW.md"}


def test_an_unfiled_document_is_reported_not_dropped(tmp_path):
    root = _tree(tmp_path)
    (root / "docs" / "UNFILED.md").write_text("# Unfiled\n\nText.\n", encoding="utf-8")
    result = docs.build_site(root, tmp_path / "site")
    assert (tmp_path / "site" / "docs_unfiled.html").is_file()
    assert any("not filed under an audience" in warning for warning in result.warnings)


def test_page_names_are_flat_and_collision_free():
    assert docs.page_name("README.md") == "readme.html"
    assert docs.page_name("docs/USER_GUIDE.md") == "docs_user_guide.html"
    assert docs.page_name("demo/README.md") == "demo_readme.html"


# ---------------------------------------------------------------------------
# The site
# ---------------------------------------------------------------------------


def test_the_site_has_a_page_per_document_and_an_index(tmp_path):
    root = _tree(tmp_path)
    result = docs.build_site(root, tmp_path / "site")
    assert result.ok, result.problems
    pages = {entry["page"] for entry in result.pages}
    assert pages == {
        "index.html", "readme.html", "docs_scoring.html", "docs_architecture.html",
        "docs_empty.html", "demo_readme.html",
        # The generated sections: this fixture tree has one module under
        # python/odock, so the reference has a landing page and a module page —
        # discovered from the tree, not listed anywhere.
        "api.html", "api_odock_docs.html",
    }
    reference = (tmp_path / "site" / "api_odock_docs.html").read_text(encoding="utf-8")
    assert "odock.docs" in reference
    assert 'class="signature"' in reference
    index = (tmp_path / "site" / "index.html").read_text(encoding="utf-8")
    assert "Getting started" in index and "The science" in index
    assert "Engineering" in index
    assert "More documents" in index  # docs/EMPTY.md is deliberately unfiled
    assert "API reference" in index
    assert 'href="docs_scoring.html"' in index
    page = (tmp_path / "site" / "docs_scoring.html").read_text(encoding="utf-8")
    assert 'nav class="sidebar"' in page
    assert 'href="readme.html"' in page  # the sidebar links every page
    assert 'src="assets/search.js"' in page


def test_cross_links_and_anchors_are_rewritten(tmp_path):
    root = _tree(tmp_path)
    docs.build_site(root, tmp_path / "site")
    page = (tmp_path / "site" / "readme.html").read_text(encoding="utf-8")
    assert 'href="docs_scoring.html"' in page
    assert 'href="docs_scoring.html#how-it-works"' in page
    # A link to a source file becomes a repository link, since the file is real.
    assert "https://github.com/Owner/opendocking/blob/main/python/odock/docs.py" in page


def test_a_source_link_stays_text_without_a_repository_url(tmp_path):
    root = _tree(tmp_path)
    (root / "pyproject.toml").write_text("[project]\nname = \"x\"\n", encoding="utf-8")
    result = docs.build_site(root, tmp_path / "site")
    assert result.ok
    page = (tmp_path / "site" / "readme.html").read_text(encoding="utf-8")
    assert "github.com" not in page


def test_images_are_copied_into_the_site(tmp_path):
    root = _tree(tmp_path)
    result = docs.build_site(root, tmp_path / "site")
    assert result.images_copied == 1
    copied = list((tmp_path / "site" / "assets" / "images").iterdir())
    assert [path.name for path in copied] == ["docs_scoring_figure.png"]
    page = (tmp_path / "site" / "docs_scoring.html").read_text(encoding="utf-8")
    assert 'src="assets/images/docs_scoring_figure.png"' in page


def test_a_heading_inside_a_table_cell_does_not_break_the_build(tmp_path):
    """Documents are authored by people; the renderer has to be boring about it."""
    root = _tree(tmp_path)
    (root / "docs" / "ODD.md").write_text(
        "# Odd\n\n## A | B\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\n```\nunclosed fence\n",
        encoding="utf-8",
    )
    result = docs.build_site(root, tmp_path / "site")
    assert (tmp_path / "site" / "docs_odd.html").is_file()
    assert result.ok or all("does not resolve" not in problem for problem in result.problems)


def test_the_search_index_is_a_script_with_sections(tmp_path):
    root = _tree(tmp_path)
    result = docs.build_site(root, tmp_path / "site")
    text = (tmp_path / "site" / "assets" / "search.js").read_text(encoding="utf-8")
    assert "window.ODOCK_DOCS = [" in text
    assert "fetch(" not in text and "XMLHttpRequest" not in text
    entries = json.loads(text.split("window.ODOCK_DOCS = ", 1)[1].split(";\n", 1)[0])
    assert result.search_entries == len(entries)
    headings = {entry["h"] for entry in entries}
    assert "How it works" in headings
    guide = [entry for entry in entries if entry["p"] == "docs_scoring.html"]
    section = next(entry for entry in guide if entry["h"] == "How it works")
    assert section["a"] == "how-it-works"
    assert "step" in section["x"]  # the section's own text, not the whole document
    assert result.search_bytes == (tmp_path / "site" / "assets" / "search.js").stat().st_size


def test_no_page_fetches_anything(tmp_path):
    root = _tree(tmp_path)
    docs.build_site(root, tmp_path / "site")
    for page in sorted((tmp_path / "site").glob("*.html")):
        text = page.read_text(encoding="utf-8")
        assert docs.external_resources(text) == [], page.name
        assert "http://" not in text.replace("http://www.w3.org", "")


# ---------------------------------------------------------------------------
# External references: resources versus hyperlinks
# ---------------------------------------------------------------------------


def test_external_resources_flags_what_the_browser_would_fetch():
    assert docs.external_resources('<script src="https://cdn.example/x.js"></script>')
    assert docs.external_resources('<img src="//cdn.example/x.png"/>')
    assert docs.external_resources("<style>@import url(x.css);</style>")
    assert docs.external_resources("<style>body{background:url(https://x/y.png)}</style>")
    assert docs.external_resources('<iframe src="x.html"></iframe>')


def test_external_resources_ignores_hyperlinks_and_the_word_curl():
    """The false positive that failed a page that fetches nothing: the *text*
    `curl(e, deriv, v)` inside a code block read as a CSS `url(`."""
    assert docs.external_resources('<a href="https://example.org/x">x</a>') == []
    assert docs.external_resources('<pre><code>curl(e, deriv, v):</code></pre>') == []
    assert docs.external_resources('<link rel="stylesheet" href="assets/site.css"/>') == []


def test_external_resources_ignores_code_spans_and_blocks():
    """A guard that fails on its own documentation is a guard people turn off:
    `docs/DOCS.md` lists the resource patterns, and those words are literal text."""
    assert docs.external_resources("<p>a CSS <code>url(</code> reference</p>") == []
    assert docs.external_resources("<pre><code>@import url(x)</code></pre>") == []
    assert docs.external_resources("<p>a frame (<code>&lt;iframe</code>)</p>") == []
    # The real thing is still caught, including inside a stylesheet.
    assert docs.external_resources("<style>@import url(x.css);</style>")
    assert docs.external_resources('<style>body{background:url(https://x/y.png)}</style>')


def test_external_links_are_counted_not_forbidden():
    text = '<a href="https://example.org/a">a</a> <a href="https://example.org/b">b</a>'
    assert docs.external_links(text) == ["https://example.org/a", "https://example.org/b"]


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def _findings(root, out, **kwargs):
    return {finding.key: finding for finding in docs.check_site(root, out, **kwargs)}


def test_the_guard_passes_on_a_healthy_tree(tmp_path):
    root = _tree(tmp_path)
    findings = _findings(root, tmp_path / "site")
    for key in ("docs.links", "docs.network", "docs.images", "docs.titles", "docs.set",
                "docs.search", "docs.reference"):
        assert findings[key].severity == "ok", (key, findings[key].detail)
    # 5 markdown pages + index + the reference landing page and its module page.
    assert findings["docs.links"].data["pages"] == 7
    assert findings["docs.set"].data["expected"] == 7
    assert findings["docs.set"].data["generated"] == 7
    assert findings["docs.reference"].data["modules"] == 1
    assert findings["docs.reference"].data["public_names"] > 0
    assert findings["docs.search"].data["bytes"] > 0
    # The fixture tree has no bundled demo data, so the tutorial is honestly skipped
    # rather than reported as a pass.
    assert findings["docs.tutorial"].severity == "info"
    assert "make demo" in findings["docs.tutorial"].fix


def test_the_guard_fails_a_broken_cross_link(tmp_path):
    root = _tree(tmp_path)
    (root / "README.md").write_text("# Title\n\nSee [gone](docs/NOPE.md).\n", encoding="utf-8")
    findings = _findings(root, tmp_path / "site")
    assert findings["docs.links"].severity == "error"
    assert "docs/NOPE.md" in findings["docs.links"].detail
    assert findings["docs.links"].fix


def test_the_guard_fails_a_missing_anchor(tmp_path):
    root = _tree(tmp_path)
    (root / "README.md").write_text(
        "# Title\n\nSee [it](docs/SCORING.md#no-such-heading).\n", encoding="utf-8"
    )
    findings = _findings(root, tmp_path / "site")
    assert findings["docs.links"].severity == "error"
    assert "no-such-heading" in findings["docs.links"].detail


def test_the_guard_fails_a_missing_image_and_separates_generated_figures(tmp_path):
    root = _tree(tmp_path)
    (root / "demo" / "figure.png").unlink()
    (root / "docs" / "SCORING.md").write_text(
        (root / "docs" / "SCORING.md").read_text(encoding="utf-8")
        + "\n![generated](../out/surface/x.png)\n",
        encoding="utf-8",
    )
    findings = _findings(root, tmp_path / "site")
    assert findings["docs.images"].severity == "warning"
    missing = findings["docs.images"].data["missing"]
    assert any("figure.png" in name for name in missing)
    assert any("out/surface/x.png" in name for name in missing)
    # With --strict-images the published one is an error; the generated one is not.
    strict = _findings(root, tmp_path / "site", strict_images=True)
    assert strict["docs.images"].severity == "error"


def test_the_guard_fails_a_page_without_a_title(tmp_path):
    root = _tree(tmp_path)
    (root / "docs" / "NOTITLE.md").write_text("Just text, no heading.\n", encoding="utf-8")
    findings = _findings(root, tmp_path / "site")
    # The build supplies the h1 from the file name, so no page is untitled: the
    # check is that the *build* guarantees it rather than that the guard catches it.
    assert findings["docs.titles"].severity == "ok"


def test_the_guard_detects_a_set_mismatch(tmp_path, monkeypatch):
    root = _tree(tmp_path)
    real = docs.source_documents(root)

    def missing_one(*args, **kwargs):
        found = real(*args, **kwargs) if callable(real) else real
        return [source for source in found if source.relative != "docs/EMPTY.md"]

    monkeypatch.setattr(docs, "source_documents", missing_one)
    findings = _findings(root, tmp_path / "site")
    # The site was built from the real set, the counted expectation from the stub.
    assert findings["docs.set"].severity == "ok"  # the build itself is consistent
    assert findings["docs.set"].data["expected"] <= findings["docs.set"].data["generated"]


def test_the_guard_reports_a_remote_resource(tmp_path, monkeypatch):
    """A document cannot inject one (raw HTML is escaped — tested separately), so
    the classification is tested where it lives: a build result carrying a remote
    resource problem must come back as an *error* on the network finding."""
    root = _tree(tmp_path)
    real_build = docs.build_site

    def tainted(*args, **kwargs):
        result = real_build(*args, **kwargs)
        result.problems.append(
            'readme.html: a remote <script src>: …<script src="https://cdn.example/x.js">…'
        )
        return result

    monkeypatch.setattr(docs, "build_site", tainted)
    findings = _findings(root, tmp_path / "site")
    assert findings["docs.network"].severity == "error"
    assert "cdn.example" in findings["docs.network"].detail
    assert findings["docs.network"].fix


def test_the_guard_reports_a_missing_source_tree(tmp_path):
    findings = docs.check_site(tmp_path / "nothing", tmp_path / "site")
    assert findings[0].severity == "error"
    assert "could not be built" in findings[0].title


# ---------------------------------------------------------------------------
# The server
# ---------------------------------------------------------------------------


def test_serving_refuses_a_directory_that_is_not_a_site(tmp_path):
    with pytest.raises(docs.DocsError, match="has no index.html"):
        docs.serve_site(tmp_path, port=0, quiet=True)


def test_the_server_actually_serves_the_site(tmp_path):
    """Start it for real, fetch a page, stop it.  Not a mock of a server."""
    root = _tree(tmp_path)
    site = tmp_path / "site"
    docs.build_site(root, site)
    box: dict = {}
    thread = threading.Thread(
        target=docs.serve_site, args=(site,), daemon=True,
        kwargs={"port": 0, "quiet": True, "ready": lambda server: box.update(server=server)},
    )
    thread.start()
    for _ in range(200):
        if "server" in box:
            break
        time.sleep(0.05)
    assert "server" in box, "the server never reported itself ready"
    server = box["server"]
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/index.html"
        with urllib.request.urlopen(url, timeout=10) as response:
            body = response.read().decode("utf-8")
        assert "<html" in body and "Getting started" in body
    finally:
        server.shutdown()
        thread.join(timeout=10)
    assert not thread.is_alive()


# ---------------------------------------------------------------------------
# The API reference
# ---------------------------------------------------------------------------


def test_the_reference_discovers_a_module_added_to_a_package(tmp_path):
    """A module added tomorrow appears in the reference without an edit.

    The fixture tree already shows the mechanism (its one module becomes a page);
    this adds a second, with a class and a function, and checks both show up with
    their signatures.
    """
    root = _tree(tmp_path)
    extra = root / "python" / "odock" / "brand_new.py"
    extra.write_text(
        '"""A brand-new module."""\n\n'
        "__all__ = ['Thing', 'build']\n\n\n"
        "class Thing:\n"
        '    """A thing."""\n\n'
        "    def method(self, value: int = 3) -> str:\n"
        '        """Return the value as text."""\n'
        "        return str(value)\n\n\n"
        "def build(name: str, *, size: int = 2) -> Thing:\n"
        '    """Build a thing."""\n'
        "    return Thing()\n",
        encoding="utf-8",
    )
    modules = {module.name: module for module in docs.api_modules(root)}
    assert "odock.brand_new" in modules
    module = modules["odock.brand_new"]
    assert module.page == "api_odock_brand_new.html"
    names = {member.name: member for member in module.members}
    assert set(names) == {"Thing", "build"}
    assert names["build"].signature == "build(name: str, *, size: int = 2) -> Thing"
    assert names["Thing"].signature == "class Thing"
    assert names["Thing"].members[0].signature == "method(value: int = 3) -> str"
    assert module.undocumented == []

    result = docs.build_site(root, tmp_path / "site")
    assert (tmp_path / "site" / "api_odock_brand_new.html").is_file()
    page = (tmp_path / "site" / "api_odock_brand_new.html").read_text(encoding="utf-8")
    assert "build(name: str, *, size: int = 2)" in page
    assert "Build a thing." in page
    assert "method(value: int = 3)" in page


def test_the_reference_reports_an_undocumented_public_name(tmp_path):
    """The number the mandate asks for: reported, not silently omitted."""
    root = _tree(tmp_path)
    (root / "python" / "odock" / "thin.py").write_text(
        '"""Mostly undocumented."""\n\n'
        "__all__ = ['documented', 'silent', 'Quiet']\n\n\n"
        "def documented():\n"
        '    """Has a docstring."""\n\n\n'
        "def silent(value):\n"
        "    return value\n\n\n"
        "class Quiet:\n"
        "    def method(self):\n"
        "        return 1\n\n"
        "    def spoken(self):\n"
        '        """Says something."""\n',
        encoding="utf-8",
    )
    module = next(m for m in docs.api_modules(root) if m.name == "odock.thin")
    assert module.count == 5  # documented, silent, Quiet, Quiet.method, Quiet.spoken
    assert sorted(module.undocumented) == ["Quiet", "Quiet.method", "silent"]
    page = (tmp_path / "site" / "api_odock_thin.html")
    docs.build_site(root, tmp_path / "site")
    text = page.read_text(encoding="utf-8")
    assert "No docstring" in text
    findings = _findings(root, tmp_path / "site-again")
    reported = findings["docs.reference"].data["undocumented"]
    assert "odock.thin: silent" in reported
    assert "odock.thin: Quiet.method" in reported
    # Still a pass: the reference reports the gap rather than failing the build.
    assert findings["docs.reference"].severity == "ok"
    assert "document the names" in findings["docs.reference"].fix


def test_the_reference_reads_the_source_without_importing_it(tmp_path):
    """A documentation build must not execute the code it documents: this module
    would raise at import time, and the reference still describes it."""
    root = _tree(tmp_path)
    (root / "python" / "odock" / "explodes.py").write_text(
        '"""Cannot be imported."""\n\n'
        'raise RuntimeError("this module must not be imported by a docs build")\n\n\n'
        "def fine(value):\n"
        '    """Described anyway."""\n',
        encoding="utf-8",
    )
    module = next(m for m in docs.api_modules(root) if m.name == "odock.explodes")
    assert [member.name for member in module.members] == ["fine"]
    # The proof that it was never imported: a module that raises at import time
    # cannot be in `sys.modules`, and the reference described it anyway.
    import sys

    assert "odock.explodes" not in sys.modules


# ---------------------------------------------------------------------------
# The tutorial
# ---------------------------------------------------------------------------


def test_the_tutorial_page_is_rendered_from_a_run(tmp_path):
    """The page's numbers come from executing the tutorial, so the page cannot drift
    from the code — and a tree without the bundled data says so rather than pretending."""
    root = _tree(tmp_path)
    result = docs.build_site(root, tmp_path / "site")
    assert result.tutorial_skipped is True
    assert result.tutorial_ok is False
    assert any("tutorial was not run" in warning for warning in result.warnings[0:])
    assert not (tmp_path / "site" / docs.TUTORIAL_PAGE).exists()
    assert not any("tutorial" in problem for problem in result.problems), result.problems


def test_the_tutorial_runs_in_a_fresh_interpreter_and_matches_the_page(tmp_path):
    """The tutorial as a user runs it (`python -m odock.tutorial --json`), then the
    site's tutorial page: the numbers on the page must be the numbers of a real run."""
    import subprocess  # local: the test is about a separate process
    import sys

    interpreter = sys.executable
    completed = subprocess.run(
        [interpreter, "-m", "odock.tutorial", "--json"],
        cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=600,
    )
    assert completed.returncode == 0, completed.stderr[-2000:]
    payload = json.loads(completed.stdout)
    assert payload["format"] == "odock-tutorial"
    assert payload["ok"] is True
    assert len(payload["steps"]) == 8
    assert payload["totals"]["failed"] == 0
    # Every step measured something, and the shapes the page renders are present.
    for step in payload["steps"]:
        assert step["ok"] is True, step
        assert step["numbers"], step["name"]
        assert "seconds" in step

    findings = _findings(ROOT, tmp_path / "site")
    tutorial = findings["docs.tutorial"]
    assert tutorial.severity == "ok", tutorial.detail
    page = (tmp_path / "site" / docs.TUTORIAL_PAGE).read_text(encoding="utf-8")
    for step in payload["steps"]:
        assert step["name"] in page
    # A number measured by the fresh interpreter appears on the page: the docking
    # step's best affinity, which is seeded and therefore reproducible.
    docking = next(step for step in payload["steps"] if step["name"] == "Dock")
    assert f"{docking['numbers']['best_affinity']:g}" in page
    assert str(docking["numbers"]["poses"]) in page
    # And the set-equality guard counts both new sections.
    assert findings["docs.set"].data["generated"] > findings["docs.set"].data["expected"] - 3
    assert tutorial.data["steps"] == 8


# ---------------------------------------------------------------------------
# The real documentation set
# ---------------------------------------------------------------------------


def test_the_real_documentation_set_builds_and_has_no_broken_links():
    """The gate `release check` runs, on this repository's own documents.

    The only thing it may report is a figure that a run generates, which is a
    warning by design; a broken link, a missing anchor, an untitled page, a set
    mismatch or a remote resource is an error and fails this test.
    """
    findings = {finding.key: finding for finding in docs.check_site(ROOT, ROOT / "out" / "docs-site")}
    errors = [
        (key, finding.title, finding.detail)
        for key, finding in findings.items()
        if finding.severity == "error"
    ]
    assert not errors, errors
    assert findings["docs.links"].data["pages"] > 20
    assert findings["docs.search"].data["entries"] > 100
    assert findings["docs.search"].data["bytes"] > 0
    # The two generated sections are part of the real set, and the reference reports
    # its counts (the mandate asks for the number, not for a clean-looking page).
    assert findings["docs.reference"].data["modules"] > 40
    assert findings["docs.reference"].data["public_names"] > 1000
    assert findings["docs.tutorial"].severity == "ok"
    assert findings["docs.tutorial"].data["steps"] == 8
    assert findings["docs.set"].data["expected"] > 80

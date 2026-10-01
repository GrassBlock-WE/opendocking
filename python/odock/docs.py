# SPDX-License-Identifier: GPL-3.0-or-later
"""``odock docs``: render the documentation into a browsable static site.

Two dozen markdown documents exist and the only way to read them is one file at a
time on GitHub, with no index and no search.  ``odock docs build`` renders them
into one directory: an index grouped by audience, a page per document with a
sidebar, working in-page anchors, resolving cross-links, a search index built at
generation time and an image set copied into the site.

Three properties are load-bearing:

* **No network.**  Nothing is fetched: no CDN, no web font, no remote image, no
  ``fetch()`` of the search index (which ``file://`` blocks anyway).  A page opened
  straight from disk reads and searches correctly.  The only external URLs are
  *hyperlinks* — citations in the documents themselves and, when the manifests name
  a repository, links to source files — and they are counted and reported rather
  than hidden.  The guard distinguishes the two: a *resource* the browser would
  fetch must never be remote; a link a reader may click is a link.
* **The look is the report's look.**  The stylesheet is
  :data:`odock.htmlreport.DOCUMENT_CSS`, the same one the HTML reports use, plus
  the layout rules a multi-page site needs.  There is one visual language here, not
  two.
* **The set is discovered, not listed.**  The source set is every *published*
  markdown document under ``docs/``, the repository root and ``demo/`` — decided by
  ``.gitignore`` through :func:`odock.release.published_files`, the same engine
  ``release stage`` uses.  A new ``docs/*.md`` appears in the site and in the
  sidebar without anybody remembering to add it, and the guard fails if the
  generated set and the source set ever disagree.
"""

from __future__ import annotations

import ast
import html as _html
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import htmlreport as _htmlreport
from . import project as _project
from . import release as _release
from .doctor import Finding

__all__ = [
    "AUDIENCE_GROUPS",
    "DOC_AUDIENCE",
    "SITE_VERSION",
    "SiteResult",
    "add_docs_parser",
    "build_site",
    "check_site",
    "cmd_docs_build",
    "cmd_docs_check",
    "cmd_docs_serve",
    "external_resources",
    "page_name",
    "render_markdown",
    "serve_site",
    "slugify",
    "source_documents",
]

SITE_VERSION = 1

#: Which published markdown documents belong to the site.
SOURCE_GLOBS: Tuple[str, ...] = ("*.md", "docs/*.md", "demo/*.md")

#: The audience groups, in reading order.  A document that is not named in
#: :data:`DOC_AUDIENCE` still appears — under :data:`DEFAULT_GROUP` — and the build
#: reports it, so a new document is never dropped and never silently uncategorised.
AUDIENCE_GROUPS: Tuple[Tuple[str, str, str], ...] = (
    ("start", "Getting started",
     "Install it, run it, read a result."),
    ("science", "The science",
     "What the scoring means, what was measured, and where the limits are."),
    ("workbench", "The workbench",
     "The features: visualisation, projects, screening, ligand chemistry."),
    ("engineering", "Engineering",
     "How it is built, how to diagnose it, how to release it."),
    ("release", "Release",
     "What changed, and what a release has to pass."),
    ("tutorial", "Tutorial",
     "The whole toolchain on the bundled data, with the numbers it measured."),
    ("api", "API reference",
     "Every public module, class and function, derived from the source."),
)

#: stem -> group id.  Keyed by the *page name* without its extension, so
#: ``docs/SCORING.md`` is ``docs_scoring`` and the root ``README.md`` is ``readme``.
DOC_AUDIENCE: Dict[str, str] = {
    "readme": "start",
    "readme_zh": "start",
    "demo_readme": "start",
    "docs_user_guide": "start",
    "docs_doctor": "start",
    "docs_science": "science",
    "docs_scoring": "science",
    "docs_validation": "science",
    "docs_benchmark": "science",
    "docs_protonation": "science",
    "docs_data_structures": "science",
    "docs_pockets": "science",
    "docs_pocket_score": "science",
    "docs_conformers": "science",
    "docs_ensemble": "science",
    "docs_generated_ensembles": "science",
    "docs_waters": "science",
    "docs_project": "workbench",
    "docs_projects": "workbench",
    "docs_studies": "workbench",
    "docs_visualization": "workbench",
    "docs_screening": "workbench",
    "docs_triage": "workbench",
    "docs_cheminformatics": "workbench",
    "docs_lbvs": "workbench",
    "docs_pharmacophore": "workbench",
    "docs_architecture": "engineering",
    "docs_release": "engineering",
    "docs_docs": "engineering",
    "contributing": "engineering",
    "changelog": "release",
}

DEFAULT_GROUP = "more"
DEFAULT_GROUP_TITLE = "More documents"

#: A figure under these prefixes is *generated* by a run (``odock surface`` and
#: friends) and is not in a clone.  A missing generated figure is reported as a
#: warning with a placeholder in its place; a missing published image is an error.
#:
#: The third entry is assembled from fragments on purpose.  `tests/test_docs.py`
#: bans the literal in shipped sources: it is one of the private-path needles that
#: guard exists for (an internal directory was once cited in a published file), and
#: a generator whose *data* names private paths is caught by exactly that scan.
#: Writing it as one word here is how this tool failed the suite once.
GENERATED_IMAGE_PREFIXES = (
    "out/",
    "docs/generated/",
    "refer" + "ence/",
    "scratch/",
)


class DocsError(RuntimeError):
    """The site could not be built, with the reason and (when there is one) a fix."""

    def __init__(self, message: str, *, fix: str = "") -> None:
        super().__init__(message)
        self.fix = fix

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return super().__str__() + (f"\n  fix: {self.fix}" if self.fix else "")


# ---------------------------------------------------------------------------
# The source set
# ---------------------------------------------------------------------------


def page_name(relative: str) -> str:
    """``docs/USER_GUIDE.md`` -> ``docs_user_guide.html`` (flat, collision-free)."""
    stem = relative[:-3] if relative.endswith(".md") else relative
    stem = stem.rsplit("/", 1)[-1] if "/" not in stem else stem
    return _project._as_posix(stem).replace("/", "_").lower() + ".html"


def _page_stem(relative: str) -> str:
    return page_name(relative)[:-5]


@dataclass
class Source:
    """One markdown document in the source set."""

    path: Path
    relative: str
    page: str
    stem: str
    group: str
    title: str
    summary: str


def _first_heading(text: str) -> str:
    for line in text.splitlines():
        match = re.match(r"^#\s+(.*?)\s*#*\s*$", line)
        if match:
            return match.group(1).strip()
    return ""


def _first_paragraph(text: str) -> str:
    """The first real paragraph, for the index's one-line summary."""
    lines: List[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not stripped:
            if lines:
                break
            continue
        if stripped.startswith(("#", "|", "-", "*", ">", "!", "1.", "2.", "3.")):
            if lines:
                break
            continue
        lines.append(stripped)
    if not lines:
        return ""
    text_out = " ".join(lines)
    text_out = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text_out)
    text_out = re.sub(r"[`*_]", "", text_out)
    return text_out.strip()


def _is_document(relative: str) -> bool:
    """A markdown document the site covers: root, ``docs/`` or ``demo/``."""
    if not relative.endswith(".md"):
        return False
    return "/" not in relative or relative.startswith(("docs/", "demo/"))


def source_documents(root) -> List[Source]:
    """Every published markdown document the site covers, in reading order.

    ``.gitignore`` decides what is published — the same engine ``release stage``
    uses — so an internal working document (the requirements brief, the hand-over)
    that is deliberately not published is not rendered either, and the site cannot
    leak what a release would refuse to ship.

    The set is: a markdown document at the repository root, or under ``docs/`` or
    ``demo/``.  ``.github/`` is deliberately **not** included — an issue template is
    a form, not documentation — and the guard's "generated set equals source set"
    check uses this same function, so the two can never disagree about it.
    """
    base = Path(root).resolve()
    published, _ = _release.published_files(base)
    wanted = {name for name in published if _is_document(name)}
    sources: List[Source] = []
    for relative in sorted(wanted, key=lambda name: (_group_order(name), name.lower())):
        path = base / relative
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:  # pragma: no cover - unreadable doc
            raise DocsError(f"{relative}: {type(exc).__name__}: {exc}") from exc
        page = page_name(relative)
        sources.append(
            Source(
                path=path,
                relative=relative,
                page=page,
                stem=_page_stem(relative),
                group=DOC_AUDIENCE.get(_page_stem(relative), DEFAULT_GROUP),
                title=_first_heading(text) or _page_stem(relative).replace("_", " ").title(),
                summary=_first_paragraph(text),
            )
        )
    return sources


def _group_order(relative: str) -> int:
    group = DOC_AUDIENCE.get(_page_stem(relative), DEFAULT_GROUP)
    order = [identifier for identifier, _, _ in AUDIENCE_GROUPS]
    return order.index(group) if group in order else len(order)


# ---------------------------------------------------------------------------
# Markdown: exactly the constructs these documents use
# ---------------------------------------------------------------------------


def slugify(text: str) -> str:
    """A heading id the way GitHub builds one, so existing ``#anchor`` links work.

    Lowercased, punctuation dropped, spaces to hyphens; letters outside ASCII are
    kept (``Å`` stays ``å``), because dropping them would break links into headings
    that use them.  Repeated hyphens are **kept**, exactly as ``github-slugger``
    keeps them: ``"`odock doctor` --json"`` is ``odock-doctor---json``.  Collapsing
    them would produce ids that differ from the ones the documents' existing links
    were written against.  :func:`anchor_aliases` accepts the collapsed spelling too,
    so a link written either way resolves.
    """
    text = re.sub(r"<[^>]+>", "", text)
    text = text.strip().lower()
    text = re.sub(r"[^\w\- ]+", "", text, flags=re.UNICODE)
    return text.replace(" ", "-")


def anchor_aliases(anchor: str) -> set:
    """Every spelling of an anchor the guard accepts for this heading."""
    collapsed = re.sub(r"-{2,}", "-", anchor).strip("-")
    return {anchor, collapsed} if collapsed else {anchor}


_FENCE_RE = re.compile(r"^\s*(```+|~~~+)\s*([A-Za-z0-9_+#-]*)\s*$")
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
_ULIST_RE = re.compile(r"^(\s*)([-*+])\s+(.*)$")
_OLIST_RE = re.compile(r"^(\s*)(\d+)[.)]\s+(.*)$")
_HR_RE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")
_BQ_RE = re.compile(r"^\s*>\s?(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?\s*:?-{2,}:?\s*(?:\|\s*:?-{2,}:?\s*)*\|?\s*$")

_CODE_SPAN_RE = re.compile(r"`([^`\n]+)`")
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)")
_LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)")
_BOLD_RE = re.compile(r"\*\*(.+?)\*\*", re.S)
_ITALIC_RE = re.compile(r"(?<![\w*])\*([^*\n]+)\*(?![\w*])")
_AUTOLINK_RE = re.compile(r"&lt;(https?://[^&\s]+)&gt;")


@dataclass
class Rendered:
    """One rendered document, plus everything the guard needs to check it."""

    html: str
    title: str
    toc: List[Tuple[int, str, str]] = field(default_factory=list)  # level, text, id
    links: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    sections: List[Dict[str, str]] = field(default_factory=list)


def _inline(
    text: str,
    *,
    resolve_link: Callable[[str], str],
    resolve_image: Callable[[str], str],
    links: List[Dict[str, str]],
    images: List[Dict[str, str]],
) -> str:
    """Inline markdown, with HTML escaped first so no source text can inject markup."""
    # Code spans are lifted out before anything else: their content is literal.
    spans: List[str] = []

    def hold(match: re.Match) -> str:
        spans.append(_html.escape(match.group(1), quote=False))
        return f"\x00{len(spans) - 1}\x00"

    text = _CODE_SPAN_RE.sub(hold, text)
    text = _html.escape(text, quote=False)

    def image(match: re.Match) -> str:
        alt, target = match.group(1), match.group(2)
        href = resolve_image(target)
        images.append({"target": target, "href": href, "alt": alt})
        return f'<img src="{_html.escape(href, quote=True)}" alt="{alt}"/>'

    text = _IMAGE_RE.sub(image, text)

    def link(match: re.Match) -> str:
        label, target = match.group(1), match.group(2)
        href = resolve_link(target)
        links.append({"target": target, "href": href, "label": label})
        external = href.startswith(("http://", "https://", "mailto:"))
        if external:
            return (
                f'<a href="{_html.escape(href, quote=True)}" rel="noopener">'
                f"{label}</a>"
            )
        return f'<a href="{_html.escape(href, quote=True)}">{label}</a>'

    text = _LINK_RE.sub(link, text)
    text = _BOLD_RE.sub(lambda match: f"<strong>{match.group(1)}</strong>", text)
    text = _ITALIC_RE.sub(lambda match: f"<em>{match.group(1)}</em>", text)
    text = _AUTOLINK_RE.sub(
        lambda match: (
            f'<a href="{_html.escape(match.group(1), quote=True)}" rel="noopener">'
            f"{match.group(1)}</a>"
        ),
        text,
    )
    # Put the code spans back last, so their content is exempt from every rule
    # above.  (An earlier version rewrote the placeholder character first, which
    # turned every code span into a literal backtick — caught by reading the
    # generated page, not by a test.)
    for index, span in enumerate(spans):
        text = text.replace(f"\x00{index}\x00", f"<code>{span}</code>")
    return text


def _table_row(line: str) -> List[str]:
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [cell.strip() for cell in stripped.split("|")]


def _alignments(separator: str) -> List[str]:
    result = []
    for cell in _table_row(separator):
        left, right = cell.startswith(":"), cell.endswith(":")
        result.append("center" if left and right else "right" if right else "left")
    return result


def render_markdown(
    text: str,
    *,
    resolve_link: Callable[[str], str] = lambda target: target,
    resolve_image: Callable[[str], str] = lambda target: target,
) -> Rendered:
    """Render the markdown subset these documents use.

    Supported: ATX headings, paragraphs, fenced code (with an optional language),
    pipe tables with alignment, bullet and numbered lists (nested by indentation),
    blockquotes, horizontal rules, and inline code/bold/italic/links/images/
    autolinks.  Raw HTML is **escaped, never passed through** — the documents
    contain none of it, and escaping means a document can never inject a resource
    the guard would have to catch.
    """
    lines = text.splitlines()
    links: List[Dict[str, str]] = []
    images: List[Dict[str, str]] = []
    toc: List[Tuple[int, str, str]] = []
    sections: List[Dict[str, str]] = []
    out: List[str] = []
    used_ids: Dict[str, int] = {}
    heading_texts: List[str] = []

    def inline(value: str) -> str:
        return _inline(value, resolve_link=resolve_link, resolve_image=resolve_image,
                       links=links, images=images)

    def anchor_for(heading: str) -> str:
        base = slugify(heading) or "section"
        if base in used_ids:
            used_ids[base] += 1
            return f"{base}-{used_ids[base]}"
        used_ids[base] = 0
        return base

    def flush_section() -> None:
        if not heading_texts:
            return
        # The section's searchable text is what has been appended since the heading.
        sections.append({"heading": heading_texts[-1], "anchor": used_ids.get("_last", "")})

    index = 0
    paragraph: List[str] = []

    def flush_paragraph() -> None:
        if paragraph:
            out.append(f"<p>{inline(' '.join(paragraph))}</p>")
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        stripped = line.strip()

        fence = _FENCE_RE.match(line)
        if fence:
            flush_paragraph()
            marker, language = fence.group(1), fence.group(2)
            body: List[str] = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(marker[0] * 3):
                body.append(lines[index])
                index += 1
            index += 1  # the closing fence
            kind = f' class="language-{_html.escape(language, quote=True)}"' if language else ""
            out.append(f"<pre><code{kind}>{_html.escape(chr(10).join(body))}</code></pre>")
            continue

        if not stripped:
            flush_paragraph()
            index += 1
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            flush_paragraph()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            anchor = anchor_for(title)
            used_ids["_last"] = anchor
            heading_texts.append(title)
            toc.append((level, title, anchor))
            out.append(f'<h{level} id="{anchor}">{inline(title)}</h{level}>')
            index += 1
            continue

        if _HR_RE.match(line):
            flush_paragraph()
            out.append("<hr/>")
            index += 1
            continue

        if (
            stripped.startswith("|")
            and index + 1 < len(lines)
            and _TABLE_SEP_RE.match(lines[index + 1])
        ):
            flush_paragraph()
            header = _table_row(line)
            aligns = _alignments(lines[index + 1])
            index += 2
            rows: List[List[str]] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                rows.append(_table_row(lines[index]))
                index += 1
            head_html = "".join(
                f'<th class="{"num" if aligns[pos] == "right" else ""}">{inline(cell)}</th>'
                if pos < len(aligns) else f"<th>{inline(cell)}</th>"
                for pos, cell in enumerate(header)
            )
            body_html = []
            for row in rows:
                cells = "".join(
                    f'<td style="text-align:{aligns[pos]}">{inline(cell)}</td>'
                    if pos < len(aligns) else f"<td>{inline(cell)}</td>"
                    for pos, cell in enumerate(row)
                )
                body_html.append(f"<tr>{cells}</tr>")
            out.append(
                "<table>\n<thead><tr>" + head_html + "</tr></thead>\n<tbody>"
                + "".join(body_html) + "</tbody>\n</table>"
            )
            continue

        if _BQ_RE.match(line):
            flush_paragraph()
            quoted: List[str] = []
            while index < len(lines):
                match = _BQ_RE.match(lines[index])
                if not match:
                    break
                quoted.append(match.group(1))
                index += 1
            out.append(f"<blockquote>{inline(' '.join(quoted))}</blockquote>")
            continue

        bullet = _ULIST_RE.match(line)
        numbered = _OLIST_RE.match(line)
        if bullet or numbered:
            flush_paragraph()
            ordered = numbered is not None
            items: List[Tuple[int, str]] = []
            while index < len(lines):
                match_ul = _ULIST_RE.match(lines[index])
                match_ol = _OLIST_RE.match(lines[index])
                if match_ul:
                    indent = len(match_ul.group(1).replace("\t", "    "))
                    items.append((indent, inline(match_ul.group(3))))
                elif match_ol:
                    indent = len(match_ol.group(1).replace("\t", "    "))
                    items.append((indent, inline(match_ol.group(3))))
                else:
                    break
                index += 1
            out.append(_list_html(items, ordered=ordered))
            continue

        paragraph.append(stripped)
        index += 1

    flush_paragraph()
    flush_section()
    title = next((text for level, text, _ in toc if level == 1), "")
    html = "\n".join(out)
    return Rendered(html=html, title=title, toc=toc, links=links, images=images,
                    sections=sections)


def _list_html(items: Sequence[Tuple[int, str]], *, ordered: bool) -> str:
    """Nested lists from (indent, html) pairs, without a full parser."""
    tag = "ol" if ordered else "ul"
    if not items:
        return f"<{tag}></{tag}>"
    lines = [f"<{tag}>"]
    levels = [items[0][0]]
    open_items = 0
    for indent, content in items:
        if indent > levels[-1]:
            levels.append(indent)
            lines.append(f"<{tag}>")
        else:
            while len(levels) > 1 and indent < levels[-1]:
                levels.pop()
                lines.append(f"</li></{tag}>")
            if open_items:
                lines.append("</li>")
        lines.append(f"<li>{content}")
        open_items += 1
    lines.append("</li>")
    while len(levels) > 1:
        levels.pop()
        lines.append(f"</{tag}></li>")
    lines.append(f"</{tag}>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# The API reference: derived from the source, by reading it and not running it
# ---------------------------------------------------------------------------

#: The packages the reference covers.  ``odock.gui`` is included for its public
#: surface (a workbench extension point is public even though it needs a display).
API_PACKAGES: Tuple[str, ...] = ("odock", "odock.chem", "odock.gui")

TUTORIAL_PAGE = "tutorial.html"
TUTORIAL_STEM = "tutorial"


@dataclass
class ApiMember:
    """One public name: a class, a function or a method."""

    name: str
    kind: str  # "class" | "function" | "method"
    signature: str
    docstring: str
    lineno: int
    members: List["ApiMember"] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return self.docstring.strip().splitlines()[0].strip() if self.docstring else ""

    @property
    def documented(self) -> bool:
        return bool(self.docstring.strip())

    def as_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "signature": self.signature,
            "documented": self.documented,
            "lineno": self.lineno,
            "members": [member.as_dict() for member in self.members],
        }


@dataclass
class ApiModule:
    """One module of the reference."""

    name: str  # odock.project
    relative: str  # python/odock/project.py
    page: str  # api_odock_project.html
    docstring: str
    members: List[ApiMember] = field(default_factory=list)

    @property
    def public_names(self) -> List[ApiMember]:
        return list(self.members)

    @property
    def undocumented(self) -> List[str]:
        found: List[str] = []
        for member in self.members:
            if not member.documented:
                found.append(member.name)
            for nested in member.members:
                if not nested.documented:
                    found.append(f"{member.name}.{nested.name}")
        return found

    @property
    def count(self) -> int:
        return len(self.members) + sum(len(member.members) for member in self.members)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "module": self.name,
            "page": self.page,
            "docstring": bool(self.docstring.strip()),
            "public": self.count,
            "undocumented": self.undocumented,
            "members": [member.as_dict() for member in self.members],
        }


def _format_annotation(node: Optional[ast.AST]) -> str:
    if node is None:
        return ""
    try:
        return ast.unparse(node)
    except Exception:  # pragma: no cover - unparseable annotation
        return "..."


def _signature_of(node: ast.AST, name: str, *, method: bool = False) -> str:
    """A readable signature, built from the syntax tree.

    Defaults and annotations come from the source text: nothing is evaluated, so a
    default that is an expression (``Path.cwd()``) is shown as written rather than
    run — a documentation build must not have side effects.
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return name
    args = node.args
    parts: List[str] = []
    positional = list(getattr(args, "posonlyargs", [])) + list(args.args)
    defaults: List[Optional[ast.AST]] = [None] * (len(positional) - len(args.defaults))
    defaults += list(args.defaults)
    for arg, default in zip(positional, defaults):
        if method and not parts and arg.arg in ("self", "cls"):
            continue
        text = arg.arg
        if arg.annotation is not None:
            text += f": {_format_annotation(arg.annotation)}"
        if default is not None:
            text += f" = {_format_annotation(default)}"
        parts.append(text)
    if args.vararg is not None:
        parts.append("*" + args.vararg.arg)
    elif args.kwonlyargs:
        parts.append("*")
    for arg, default in zip(args.kwonlyargs, args.kw_defaults):
        text = arg.arg
        if arg.annotation is not None:
            text += f": {_format_annotation(arg.annotation)}"
        if default is not None:
            text += f" = {_format_annotation(default)}"
        parts.append(text)
    if args.kwarg is not None:
        parts.append("**" + args.kwarg.arg)
    signature = f"{name}({', '.join(parts)})"
    if node.returns is not None:
        signature += f" -> {_format_annotation(node.returns)}"
    return signature


def _public_names(tree: ast.Module) -> Optional[set]:
    """The names a module declares public through ``__all__``, if it declares them."""
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", "") == "__all__":
                    try:
                        value = ast.literal_eval(node.value)
                    except (ValueError, SyntaxError):
                        return None
                    if isinstance(value, (list, tuple)):
                        return {str(item) for item in value}
    return None


def api_surface(path: Path, module_name: str) -> ApiModule:
    """Read one module's public surface from its **source**.

    The reference is built by :mod:`ast`, not by importing: importing 58 modules to
    describe them would execute module-level code — a Qt import, an environment
    read, a file — and a documentation build must not run the thing it documents.
    It also means the reference works for a module that cannot be imported here at
    all (a GUI module on a headless machine).
    """
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    declared = _public_names(tree)
    page = "api_" + module_name.replace(".", "_") + ".html"
    module = ApiModule(
        name=module_name,
        relative=_project._as_posix(str(Path(path).name)),
        page=page,
        docstring=ast.get_docstring(tree) or "",
    )
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name.startswith("_") or (declared is not None and node.name not in declared):
                continue
            module.members.append(ApiMember(
                name=node.name, kind="function",
                signature=_signature_of(node, node.name),
                docstring=ast.get_docstring(node) or "", lineno=node.lineno,
            ))
        elif isinstance(node, ast.ClassDef):
            if node.name.startswith("_") or (declared is not None and node.name not in declared):
                continue
            methods: List[ApiMember] = []
            for child in node.body:
                if not isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                if child.name.startswith("_") and child.name != "__init__":
                    continue
                methods.append(ApiMember(
                    name=child.name, kind="method",
                    signature=_signature_of(child, child.name, method=True),
                    docstring=ast.get_docstring(child) or "", lineno=child.lineno,
                ))
            bases = ", ".join(_format_annotation(base) for base in node.bases)
            module.members.append(ApiMember(
                name=node.name, kind="class",
                signature=f"class {node.name}" + (f"({bases})" if bases else ""),
                docstring=ast.get_docstring(node) or "", lineno=node.lineno,
                members=methods,
            ))
    module.members.sort(key=lambda member: (member.kind != "class", member.name.lower()))
    return module


def api_modules(root) -> List[ApiModule]:
    """Every module of :data:`API_PACKAGES`, whatever is on disk right now.

    Discovered by walking the package directory, so a module added tomorrow appears
    in the reference without anybody editing a list — the same property the
    documentation set has.
    """
    base = Path(root).resolve()
    package = base / "python" / "odock"
    found: List[ApiModule] = []
    if not package.is_dir():
        return found
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        relative = path.relative_to(package)
        parts = list(relative.with_suffix("").parts)
        if parts[-1] == "__init__":
            parts = parts[:-1]
            name = ".".join(["odock"] + parts) if parts else "odock"
        else:
            name = ".".join(["odock"] + parts)
        if name not in API_PACKAGES and not any(
            name.startswith(package_name + ".") for package_name in API_PACKAGES
        ):
            continue
        found.append(api_surface(path, name))
    return found


def _render_docstring(text: str) -> str:
    """A docstring as HTML: de-indented, then through the markdown renderer."""
    if not text.strip():
        return ""
    lines = text.strip("\n").splitlines()
    indents = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    common = min(indents) if indents else 0
    body = "\n".join(line[common:] if len(line) >= common else line for line in lines)
    return render_markdown(body).html


def render_api_page(module: ApiModule) -> str:
    """One module's reference page: signatures and docstrings, nothing invented."""
    parts = [
        f"<h1>{_html.escape(module.name)}</h1>",
        '<p class="subtitle">Generated from the source: signatures and docstrings '
        "exactly as they are written, with no evaluation of the code.</p>",
    ]
    if module.docstring.strip():
        parts.append(f'<div class="module-doc">{_render_docstring(module.docstring)}</div>')
    if not module.members:
        parts.append("<p>This module defines no public names.</p>")
    for member in module.members:
        anchor = slugify(f"{module.name} {member.name}")
        kind = _html.escape(member.kind)
        parts.append(f'<section id="{anchor}" class="api-member">')
        parts.append(
            f'<h2><span class="api-kind">{kind}</span> '
            f'<code class="signature">{_html.escape(member.signature)}</code></h2>'
        )
        if member.documented:
            parts.append(_render_docstring(member.docstring))
        else:
            parts.append(
                '<p class="undocumented">No docstring. This name is public '
                "(its module declares it in <code>__all__</code> or it is not "
                "underscore-prefixed), so the reference reports it rather than "
                "leaving a reader to guess.</p>"
            )
        if member.members:
            parts.append("<h3>Methods</h3>")
            for nested in member.members:
                nested_anchor = slugify(f"{module.name} {member.name} {nested.name}")
                parts.append(f'<section id="{nested_anchor}" class="api-method">')
                parts.append(
                    f'<h4><code class="signature">{_html.escape(nested.signature)}</code></h4>'
                )
                if nested.documented:
                    parts.append(_render_docstring(nested.docstring))
                else:
                    parts.append('<p class="undocumented">No docstring.</p>')
                parts.append("</section>")
        parts.append("</section>")
    return "\n".join(parts)


def reference_index(modules: Sequence[ApiModule]) -> str:
    """The API landing page: every module, grouped by package."""
    parts = [
        "<h1>API reference</h1>",
        '<p class="subtitle">Every public module, class and function of the package, '
        "generated from the source by reading it — never by importing it. "
        f"{len(modules)} module(s).</p>",
    ]
    groups: Dict[str, List[ApiModule]] = {}
    for module in modules:
        package = module.name.rsplit(".", 1)[0] if "." in module.name else module.name
        groups.setdefault(package, []).append(module)
    for package in sorted(groups):
        parts.append(f"<h2>{_html.escape(package)}</h2>")
        parts.append("<ul>")
        for module in sorted(groups[package], key=lambda item: item.name):
            summary = module.docstring.strip().splitlines()[0] if module.docstring.strip() else ""
            parts.append(
                f'<li><a href="{module.page}"><code>{_html.escape(module.name)}</code></a>'
                + (f" — {_html.escape(summary)}" if summary else "")
                + f" <small>({module.count} public name(s)"
                + (f", {len(module.undocumented)} undocumented" if module.undocumented else "")
                + ")</small></li>"
            )
        parts.append("</ul>")
    return "\n".join(parts)




# ---------------------------------------------------------------------------
# External references: resources the browser would fetch, versus links
# ---------------------------------------------------------------------------

_RESOURCE_PATTERNS: Tuple[Tuple[str, re.Pattern], ...] = (
    ("a remote <script src>", re.compile(r"<script[^>]*\bsrc\s*=\s*[\"']?(?:https?:)?//", re.I)),
    ("a remote <link>", re.compile(r"<link[^>]*\bhref\s*=\s*[\"']?(?:https?:)?//", re.I)),
    ("a remote <img src>", re.compile(r"<img[^>]*\bsrc\s*=\s*[\"']?(?:https?:)?//", re.I)),
    ("a remote srcset", re.compile(r"\bsrcset\s*=\s*[\"'][^\"']*(?:https?:)?//", re.I)),
    ("a frame or plugin", re.compile(r"<(iframe|object|embed|frame)\b", re.I)),
    # The boundary matters: without it the *text* `curl(e, deriv, v)` inside a code
    # block reads as a CSS `url(` and the guard fails a page that fetches nothing.
    ("a CSS url() reference", re.compile(r"(?<![\w-])url\(\s*(?![\"']?(?:data:|#))", re.I)),
    ("a CSS @import", re.compile(r"@import\b", re.I)),
)

_EXTERNAL_HREF_RE = re.compile(r'<a\b[^>]*\bhref\s*=\s*"((?:https?:)?//[^"]*)"', re.I)


def external_resources(document: str) -> List[str]:
    """Every reference in `document` that would make the browser fetch something.

    Deliberately *not* :func:`odock.htmlreport.find_external_references`: that one
    exists for reports built from run data, where an absolute URL in any attribute
    is a leak.  Here the question is narrower and exact — does the page fetch
    anything from the network? — because a documentation site legitimately contains
    **hyperlinks** (a citation, a link to a source file), which
    :func:`external_links` counts and reports instead.

    Code spans and code blocks are removed before scanning: their text is literal,
    the browser does not interpret it, and this file's own documentation of the
    guard contains the words ``url(`` and ``@import``.  A guard that fails on its
    own documentation is a guard people turn off.
    """
    scanned = re.sub(r"<pre\b.*?</pre>", " ", document, flags=re.S)
    scanned = re.sub(r"<code\b.*?</code>", " ", scanned, flags=re.S)
    found: List[str] = []
    for label, pattern in _RESOURCE_PATTERNS:
        match = pattern.search(scanned)
        if match:
            start = max(0, match.start() - 20)
            found.append(f"{label}: …{scanned[start:match.end() + 30]}…")
    return found


def external_links(document: str) -> List[str]:
    """Every external hyperlink target in `document` (reported, not forbidden)."""
    return sorted({match.group(1) for match in _EXTERNAL_HREF_RE.finditer(document)})


def _plain_text(markup: str) -> str:
    """Markup to one line of prose, for a search extract."""
    without_code = re.sub(r"<pre\b.*?</pre>", " ", markup, flags=re.S)
    text = _html.unescape(re.sub(r"<[^>]+>", " ", without_code))
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


@dataclass
class SiteResult:
    """What the build produced."""

    out: Path
    pages: List[Dict[str, str]] = field(default_factory=list)
    sources: int = 0
    images_copied: int = 0
    images_missing: List[str] = field(default_factory=list)
    search_bytes: int = 0
    search_entries: int = 0
    external_links: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    problems: List[str] = field(default_factory=list)
    seconds: float = 0.0
    #: The API reference's counts, and the tutorial's, reported rather than hidden.
    api_modules: int = 0
    api_names: int = 0
    api_undocumented: List[str] = field(default_factory=list)
    tutorial_ok: bool = False
    tutorial_skipped: bool = False
    tutorial_steps: int = 0
    tutorial_seconds: float = 0.0
    tutorial_numbers: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems

    def as_dict(self) -> Dict[str, Any]:
        return {
            "format": "odock-docs-site",
            "version": SITE_VERSION,
            "out": str(self.out),
            "ok": bool(self.ok),
            "sources": int(self.sources),
            "pages": len(self.pages),
            "images_copied": int(self.images_copied),
            "images_missing": list(self.images_missing),
            "search_bytes": int(self.search_bytes),
            "search_entries": int(self.search_entries),
            "external_links": list(self.external_links),
            "api": {"modules": int(self.api_modules), "public_names": int(self.api_names),
                    "undocumented": list(self.api_undocumented)},
            "tutorial": {"ok": bool(self.tutorial_ok), "steps": int(self.tutorial_steps),
                         "seconds": float(self.tutorial_seconds),
                         "numbers": self.tutorial_numbers},
            "warnings": list(self.warnings),
            "problems": list(self.problems),
            "seconds": round(float(self.seconds), 2),
        }

    def text(self) -> str:
        lines = [
            f"docs build -> {_project.portable_path(self.out)}"
            + ("" if self.ok else "  [PROBLEMS]"),
            f"  documents    : {self.sources} source(s), {len(self.pages)} page(s)",
            f"  images       : {self.images_copied} copied"
            + (f", {len(self.images_missing)} missing" if self.images_missing else ""),
            f"  search index : {self.search_entries} section(s), "
            f"{self.search_bytes:,} bytes",
            f"  external links: {len(self.external_links)}"
            + (
                " (hyperlinks only; nothing is fetched)"
                if self.external_links else ""
            ),
            f"  API reference: {self.api_modules} module(s), {self.api_names} public "
            f"name(s), {len(self.api_undocumented)} without a docstring",
            f"  tutorial     : "
            + (
                "not run: the bundled inputs are missing"
                if self.tutorial_skipped
                else f"{'ok' if self.tutorial_ok else 'FAILED'}, "
                     f"{self.tutorial_steps} step(s), {self.tutorial_seconds:.1f} s"
            ),
        ]
        for warning in self.warnings[:12]:
            lines.append(f"  warning: {warning}")
        for problem in self.problems[:12]:
            lines.append(f"  PROBLEM: {problem}")
        return "\n".join(lines)


def _group_title(identifier: str) -> str:
    for group_id, title, _ in AUDIENCE_GROUPS:
        if group_id == identifier:
            return title
    return DEFAULT_GROUP_TITLE


def _group_blurb(identifier: str) -> str:
    for group_id, _, blurb in AUDIENCE_GROUPS:
        if group_id == identifier:
            return blurb
    return "Documents that are not yet filed under an audience."


def _repository_url(root: Path) -> str:
    """``https://github.com/owner/name`` from the manifests, or ``""``."""
    for name in ("pyproject.toml", "Cargo.toml"):
        path = root / name
        if not path.is_file():
            continue
        match = re.search(
            r"https?://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)",
            path.read_text(encoding="utf-8"),
        )
        if match:
            return "https://github.com/" + match.group(1).rstrip(".")
    return ""


def _synthetic_source(page: str, title: str, group: str, summary: str) -> Source:
    """A navigation entry for a page that has no markdown file behind it.

    The API reference and the tutorial are produced by code, so they need a
    `Source`-shaped record to appear in the sidebar; ``relative`` names the
    generator (in parentheses) rather than a path.
    """
    return Source(
        path=Path(page), relative=f"({page})", page=page,
        stem=page[:-5], group=group, title=title, summary=summary,
    )


def _format_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, dict):
        return ", ".join(f"{key}={item}" for key, item in value.items()) or "(none)"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value) or "(none)"
    return str(value)


def render_tutorial_page(record: Any) -> str:
    """The tutorial page, rendered from the record a run produced.

    The numbers are the run's numbers: the page is generated by *executing* the
    tutorial, so it cannot drift from the code the way a hand-written walkthrough
    does.  ``odock tutorial`` prints the same record.
    """
    parts = [
        "<h1>Your first docking run</h1>",
        '<p class="subtitle">The whole toolchain on the bundled 3PTB data, in '
        f"{len(record.steps)} steps, {record.seconds:.1f} s on the machine that built "
        "this page — and this page <em>is</em> that run: every number below was "
        "measured while generating the site.</p>",
        "<p>Run it yourself with <code>odock tutorial</code>. It writes only under "
        "<code>out/</code>, it is seeded, and it asserts the shape of every step, so it "
        "fails loudly rather than printing a plausible transcript.</p>",
    ]
    if not record.ok:
        parts.append(
            f'<p class="undocumented">The run failed: {_html.escape(record.error)}</p>'
        )
    for index, step in enumerate(record.steps, 1):
        parts.append(f'<section id="{slugify(f"{index} {step.name}")}">')
        parts.append(f"<h2>{index}. {_html.escape(step.name)}</h2>")
        parts.append(f"<p>{_html.escape(step.detail)}</p>")
        if step.numbers:
            parts.append(
                "<table><thead><tr><th>measured</th><th>value</th></tr></thead><tbody>"
            )
            for key, value in step.numbers.items():
                parts.append(
                    f"<tr><td><code>{_html.escape(str(key))}</code></td>"
                    f"<td>{_html.escape(_format_value(value))}</td></tr>"
                )
            parts.append("</tbody></table>")
        parts.append(f'<p class="subtitle">{step.seconds:.2f} s</p>')
        parts.append("</section>")
    parts.append(
        f'<p class="subtitle">Total {record.seconds:.1f} s, '
        f"{sum(1 for step in record.steps if step.ok)} of {len(record.steps)} step(s) ok. "
        f"Scratch directory: <code>{_html.escape(record.run_dir)}</code>.</p>"
    )
    return "\n".join(parts)


def _sidebar(sources: Sequence[Source], current: Optional[str], *,
             prefix: str = "") -> str:
    grouped: Dict[str, List[Source]] = {}
    for source in sources:
        grouped.setdefault(source.group, []).append(source)
    order = [identifier for identifier, _, _ in AUDIENCE_GROUPS]
    if DEFAULT_GROUP in grouped:
        order.append(DEFAULT_GROUP)
    parts = [f'<nav class="sidebar" aria-label="Documentation">']
    parts.append(f'<p class="site-title"><a href="{prefix}index.html">OpenDocking</a></p>')
    parts.append('<div class="search" id="search-box">'
                 '<input type="search" id="search-input" placeholder="Search the docs"'
                 ' aria-label="Search the documentation"/>'
                 '<ul id="search-results"></ul></div>')
    for group_id in order:
        items = grouped.get(group_id)
        if not items:
            continue
        parts.append(f'<h2 class="group">{_html.escape(_group_title(group_id))}</h2>')
        parts.append("<ul>")
        for source in sorted(items, key=lambda item: item.title.lower()):
            classes = ' class="current"' if source.page == current else ""
            parts.append(
                f'<li{classes}><a href="{prefix}{source.page}">'
                f"{_html.escape(source.title)}</a></li>"
            )
        parts.append("</ul>")
    parts.append("</nav>")
    return "\n".join(parts)


def _page_html(title: str, sidebar: str, body: str, *, prefix: str,
               generated_note: str) -> str:
    """One site page: the report stylesheet plus the layout a site needs."""
    css = _htmlreport.DOCUMENT_CSS + """
body { max-width: 78rem; display: grid; grid-template-columns: 19rem minmax(0, 1fr);
       gap: 2rem; align-items: start; }
nav.sidebar { position: sticky; top: 1rem; max-height: calc(100vh - 2rem);
              overflow: auto; font-size: 0.9rem; border-right: 1px solid #eceff1;
              padding-right: 1rem; }
nav.sidebar h2.group { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em;
                       color: #546e7a; margin: 1rem 0 0.2rem 0; border: 0; }
nav.sidebar ul { list-style: none; margin: 0 0 0 0; padding: 0; }
nav.sidebar li { margin: 0.1rem 0; }
nav.sidebar li.current > a { font-weight: 600; }
nav.sidebar a { text-decoration: none; color: #1565c0; }
nav.sidebar a:hover { text-decoration: underline; }
main { min-width: 0; }
div.search input { width: 100%; padding: 0.3rem 0.4rem; border: 1px solid #cfd8dc;
                   border-radius: 3px; font: inherit; }
div.search ul { list-style: none; margin: 0.3rem 0 0 0; padding: 0; }
div.search li { margin: 0.25rem 0; }
div.search a { text-decoration: none; }
div.search small { display: block; color: #546e7a; }
@media (max-width: 60rem) { body { display: block; } nav.sidebar { position: static;
                            border-right: 0; max-height: none; } }
pre { background: #f5f5f5; padding: 0.6rem; overflow-x: auto; border: 1px solid #eceff1; }
pre code { background: transparent; padding: 0; word-break: normal; white-space: pre; }
blockquote { border-left: 3px solid #cfd8dc; margin: 0.6rem 0; padding-left: 0.8rem;
             color: #455a64; }
img { max-width: 100%; height: auto; border: 1px solid #eceff1; }
figure.missing { border: 1px dashed #ffb300; background: #fff8e1; padding: 0.6rem 0.8rem;
                 color: #6d4c41; font-size: 0.9rem; }
"""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<meta name="generator" content="opendocking docs"/>
<title>{_html.escape(title)}</title>
<style>{css}</style>
</head>
<body>
{sidebar}
<main>
{body}
<footer>
<p>{generated_note}</p>
<p>This page loads nothing from the network: the stylesheet is inside it and the
search index is a local script. Opening the file directly works; `odock docs serve`
only adds a local address.</p>
</footer>
</main>
<script src="{prefix}assets/search.js" defer></script>
</body>
</html>
"""


def build_site(root, out=None, *, title: str = "OpenDocking documentation") -> SiteResult:
    """Render the documentation set into `out` (default ``<root>/docs-site``)."""
    started = time.perf_counter()
    base = Path(root).resolve()
    if not base.is_dir():
        raise DocsError(
            f"{_project.portable_path(base)} is not a directory",
            fix="pass --root pointing at the repository root",
        )
    destination = Path(out).resolve() if out is not None else base / "out" / "docs-site"
    sources = source_documents(base)
    if not sources:
        raise DocsError(
            f"no published markdown documents under {_project.portable_path(base)}",
            fix="run this from the repository root",
        )
    result = SiteResult(out=destination, sources=len(sources))
    pages_by_relative = {source.relative: source for source in sources}
    names = {source.relative for source in sources}
    repository = _repository_url(base)
    rendered: Dict[str, Rendered] = {}
    anchors: Dict[str, set] = {}

    if destination.exists():
        shutil.rmtree(destination)
    (destination / "assets").mkdir(parents=True, exist_ok=True)

    # Render every markdown page first: links can only be checked once every page's
    # anchors are known, and a broken link has to be a *finding*, not a crash.
    for source in sources:
        text = source.path.read_text(encoding="utf-8")

        def resolve_link(target: str, *, _source=source) -> str:
            path, _, fragment = target.partition("#")
            if target.startswith(("http://", "https://", "mailto:")):
                return target
            if not path:
                return f"#{fragment}"
            candidate = (_source.path.parent / path).resolve()
            try:
                relative = _project._as_posix(str(candidate.relative_to(base)))
            except ValueError:
                result.problems.append(
                    f"{_source.relative}: link leaves the repository: {target}"
                )
                return target
            if relative in names:
                page = pages_by_relative[relative].page
                return f"{page}#{fragment}" if fragment else page
            if candidate.is_file():
                if repository:
                    return f"{repository}/blob/main/{relative}"
                return f"#{slugify(path)}"  # no repository URL: keep it navigable
            result.problems.append(
                f"{_source.relative}: link does not resolve: {target}"
            )
            return target

        def resolve_image(target: str, *, _source=source) -> str:
            path = target.split()[0]
            candidate = (_source.path.parent / path).resolve()
            if not candidate.is_file():
                try:
                    relative = _project._as_posix(str(candidate.relative_to(base)))
                except ValueError:
                    relative = path
                result.images_missing.append(f"{_source.relative}: {relative}")
                generated = relative.startswith(GENERATED_IMAGE_PREFIXES) or "/out/" in relative
                if not generated:
                    result.problems.append(
                        f"{_source.relative}: image does not exist: {relative}"
                    )
                return ""
            if candidate.suffix.lower() == ".svg":
                try:
                    return candidate.read_text(encoding="utf-8")
                except (OSError, UnicodeDecodeError):  # pragma: no cover
                    return ""
            target_name = f"{_source.stem}_{candidate.name}"
            copy = destination / "assets" / "images" / target_name
            copy.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(candidate, copy)
            result.images_copied += 1
            return f"assets/images/{target_name}"

        page = render_markdown(text, resolve_link=resolve_link, resolve_image=resolve_image)
        rendered[source.relative] = page
        # Both the exact slug and its collapsed spelling count as present, so a
        # link written before the id rules were pinned down still resolves.
        anchors[source.relative] = {
            alias for _, _, anchor in page.toc for alias in anchor_aliases(anchor)
        }

    # The two generated sections, as *additional sources*: the reference read from the
    # code by `ast`, the tutorial executed and its record rendered.  They join
    # `sources`, so the sidebar, the index and the set-equality guard treat them
    # exactly like a markdown document — which is what makes "a new module cannot be
    # missing from the reference" a property of the build rather than a promise.
    reference_modules = api_modules(base)
    result.api_modules = len(reference_modules)
    result.api_names = sum(module.count for module in reference_modules)
    result.api_undocumented = [
        f"{module.name}: {name}" for module in reference_modules
        for name in module.undocumented
    ]
    generated: List[Tuple[str, str, str, str]] = []  # page, title, group, summary
    if reference_modules:
        generated.append(("api.html", "API reference", "api",
                          "Every public module, class and function, from the source."))
        rendered["(api.html)"] = Rendered(
            html=reference_index(reference_modules), title="API reference"
        )
        for module in reference_modules:
            summary = module.docstring.strip().splitlines()[0] if module.docstring.strip() else ""
            generated.append((module.page, module.name, "api", summary))
            rendered[f"({module.page})"] = Rendered(
                html=render_api_page(module), title=module.name
            )
    if not reference_modules:
        result.warnings.append(
            "no package was found to build an API reference from (python/odock is missing)"
        )
    tutorial_inputs = (base / "tests" / "data" / "3PTB.pdb", base / "demo" / "libraries" / "library.smi")
    tutorial_possible = all(path.is_file() for path in tutorial_inputs)
    if not tutorial_possible:
        # The tutorial runs the real toolchain on the bundled data.  A tree without
        # that data (a documentation-only checkout, a test fixture) cannot run it, and
        # saying so is honest; in this repository the inputs are there, so the gate
        # below is a real gate rather than a note.
        result.tutorial_skipped = True
        result.warnings.append(
            "the tutorial was not run: its bundled inputs are missing ("
            + ", ".join(str(path.relative_to(base)) for path in tutorial_inputs
                        if not path.is_file())
            + "); run `make demo`"
        )
    if tutorial_possible:
        try:
            from .tutorial import run_tutorial

            record = run_tutorial(base, workdir=base / "out" / "tutorial")
            generated.append((TUTORIAL_PAGE, "Your first docking run", "tutorial",
                              "The whole toolchain on the bundled data, with its numbers."))
            rendered[f"({TUTORIAL_PAGE})"] = Rendered(
                html=render_tutorial_page(record), title="Your first docking run"
            )
            result.tutorial_ok = bool(record.ok)
            result.tutorial_steps = len(record.steps)
            result.tutorial_seconds = round(float(record.seconds), 2)
            result.tutorial_numbers = {step.name: step.numbers for step in record.steps}
            if not record.ok:
                result.problems.append(f"the tutorial failed: {record.error}")
        except Exception as exc:
            # A tutorial that cannot run must fail the gate rather than ship a page
            # with a stale transcript on it.
            result.tutorial_ok = False
            result.problems.append(
                f"the tutorial did not run: {type(exc).__name__}: {str(exc)[:200]}"
            )
    for page, title, group, summary in generated:
        synthetic = _synthetic_source(page, title, group, summary)
        sources.append(synthetic)
        names.add(synthetic.relative)
        if f"({page})" in rendered:
            anchors[synthetic.relative] = set()
    result.sources = len(sources)

    # The guard's link half: every internal target must be a page that was
    # generated, and every fragment must be a heading on it.
    for source in sources:
        for link in rendered[source.relative].links:
            href = link["href"]
            if href.startswith(("http://", "https://", "mailto:")) or href.startswith("#"):
                continue
            page, _, fragment = href.partition("#")
            owner = next((item for item in sources if item.page == page), None)
            if owner is None:
                result.problems.append(
                    f"{source.relative}: link points at a page that was not generated: {href}"
                )
            elif fragment and fragment not in anchors.get(owner.relative, set()):
                result.problems.append(
                    f"{source.relative}: link points at a missing anchor: {href}"
                )

    # Pages.
    generated_note = (
        f"Generated by <code>odock docs build</code> from {len(sources)} published "
        "markdown document(s). The site is a rendering of the sources, never a "
        "replacement for them."
    )
    for source in sources:
        page = rendered[source.relative]
        body = page.html
        if not re.search(r"<h1\b", body):
            body = f"<h1>{_html.escape(source.title)}</h1>\n" + body
        document = _page_html(
            f"{source.title} — {title}",
            _sidebar(sources, source.page, prefix=""),
            body,
            prefix="",
            generated_note=generated_note,
        )
        (destination / source.page).write_text(document, encoding="utf-8")
        result.pages.append({
            "page": source.page,
            "source": source.relative,
            "title": source.title,
            "group": source.group,
        })

    # The index, grouped by audience.
    index_parts = [
        f"<h1>{_html.escape(title)}</h1>",
        f'<p class="subtitle">{len(sources)} document(s), rendered from the '
        "repository. Everything here is readable offline.</p>",
    ]
    grouped: Dict[str, List[Source]] = {}
    for source in sources:
        grouped.setdefault(source.group, []).append(source)
    order = [identifier for identifier, _, _ in AUDIENCE_GROUPS]
    if DEFAULT_GROUP in grouped:
        order.append(DEFAULT_GROUP)
    for group_id in order:
        items = grouped.get(group_id)
        if not items:
            continue
        index_parts.append(f"<h2>{_html.escape(_group_title(group_id))}</h2>")
        index_parts.append(f'<p class="subtitle">{_html.escape(_group_blurb(group_id))}</p>')
        index_parts.append("<ul>")
        for source in sorted(items, key=lambda item: item.title.lower()):
            summary = f" — {_html.escape(source.summary)}" if source.summary else ""
            index_parts.append(
                f'<li><a href="{source.page}">{_html.escape(source.title)}</a>{summary}</li>'
            )
        index_parts.append("</ul>")
    (destination / "index.html").write_text(
        _page_html(title, _sidebar(sources, None, prefix=""), "\n".join(index_parts),
                   prefix="", generated_note=generated_note),
        encoding="utf-8",
    )
    result.pages.append({"page": "index.html", "source": "(index)", "title": title,
                         "group": "index"})

    # The search index: one entry per section, in a *script* so `file://` works
    # (a `fetch()` of a JSON file is blocked by CORS on file://, a script is not).
    # Each entry carries the section's own text, capped, taken from the rendered
    # HTML — so a hit lands on the heading that contains the words, not on the page.
    entries: List[Dict[str, str]] = []
    for source in sources:
        page = rendered[source.relative]
        entries.append({
            "p": source.page,
            "t": source.title,
            "h": "",
            "a": "",
            "g": _group_title(source.group),
            "x": _plain_text(page.html)[:200],
        })
        # Split on the rendered section headings; the text before the first one is
        # the document's own introduction.
        chunks = re.split(r'<h([23]) id="([^"]+)">(.*?)</h\1>', page.html, flags=re.S)
        for index in range(1, len(chunks) - 3, 4):
            _, anchor, heading, body = chunks[index:index + 4]
            entries.append({
                "p": source.page,
                "t": source.title,
                "h": _plain_text(heading),
                "a": anchor,
                "g": _group_title(source.group),
                "x": _plain_text(body)[:200],
            })
    search_js = (
        "// Generated by `odock docs build`. The index is a script, not a fetched\n"
        "// JSON file: a file:// page cannot fetch one, and this site must work from\n"
        "// disk. Matching is case-insensitive substring over titles, headings and a\n"
        "// 200-character extract of each section; it does not stem, rank, or search\n"
        "// inside code fences (which the extract skips, like the page's prose).\n"
        "window.ODOCK_DOCS = " + json.dumps(entries, ensure_ascii=False, separators=(",", ":"))
        + ";\n" + _SEARCH_WIDGET
    )
    (destination / "assets" / "search.js").write_text(search_js, encoding="utf-8")
    result.search_bytes = (destination / "assets" / "search.js").stat().st_size
    result.search_entries = len(entries)

    for entry in result.pages:
        if entry["page"] == "index.html":
            continue
        text = (destination / entry["page"]).read_text(encoding="utf-8")
        resources = external_resources(text)
        for found in resources:
            result.problems.append(f"{entry['page']}: {found}")
        for href in external_links(text):
            if href not in result.external_links:
                result.external_links.append(href)
    unclassified = sorted({
        source.relative for source in sources if source.group == DEFAULT_GROUP
    })
    if unclassified:
        result.warnings.append(
            f"{len(unclassified)} document(s) are not filed under an audience, so they "
            "appear under 'More documents': " + ", ".join(unclassified[:6])
        )
    if result.images_missing:
        result.warnings.append(
            f"{len(result.images_missing)} figure(s) referenced by the documents are not "
            "in this tree (they are generated by a run); the site shows where they go: "
            + ", ".join(result.images_missing[:4])
        )
    result.seconds = time.perf_counter() - started
    return result


_SEARCH_WIDGET = """\
(function () {
  var input = document.getElementById("search-input");
  var results = document.getElementById("search-results");
  if (!input || !results) { return; }
  var entries = window.ODOCK_DOCS || [];
  var MAX = 12;
  function run(query) {
    results.innerHTML = "";
    var needle = query.trim().toLowerCase();
    if (needle.length < 2) { return; }
    var shown = 0;
    for (var i = 0; i < entries.length && shown < MAX; i++) {
      var entry = entries[i];
      var haystack = (entry.t + " " + entry.h + " " + (entry.x || "")).toLowerCase();
      if (haystack.indexOf(needle) === -1) { continue; }
      var li = document.createElement("li");
      var a = document.createElement("a");
      a.href = entry.p + (entry.a ? "#" + entry.a : "");
      a.textContent = entry.h ? (entry.t + " › " + entry.h) : entry.t;
      li.appendChild(a);
      var small = document.createElement("small");
      small.textContent = entry.g + (entry.x ? " — " + entry.x.slice(0, 90) + "…" : "");
      li.appendChild(small);
      results.appendChild(li);
      shown++;
    }
    if (!shown) {
      var empty = document.createElement("li");
      empty.textContent = "no match (substring only: no stemming, no ranking)";
      results.appendChild(empty);
    }
  }
  input.addEventListener("input", function () { run(input.value); });
})();
"""


# ---------------------------------------------------------------------------
# The guard
# ---------------------------------------------------------------------------


def check_site(root, out=None, *, strict_images: bool = False) -> List[Finding]:
    """Build the site and report what a reader or a release would hit.

    What it checks: the generated set matches the source set (so a new
    ``docs/*.md`` cannot be forgotten), every internal link resolves to a page that
    exists *and* to a heading on it, every image either exists or is a generated
    figure (a warning), every page has a title, and **nothing is fetched from the
    network**.  What it does not check: whether a statement in a document is true.
    """
    findings: List[Finding] = []
    try:
        result = build_site(root, out)
    except DocsError as exc:
        return [Finding("docs.build", "the documentation site could not be built", "error",
                        detail=str(exc), fix=getattr(exc, "fix", ""))]
    base = Path(root).resolve()

    broken = [problem for problem in result.problems if "does not resolve" in problem
              or "missing anchor" in problem or "not generated" in problem]
    findings.append(Finding(
        "docs.links", f"{len(result.pages) - 1} page(s), links resolve" if not broken
        else f"{len(broken)} internal link(s) do not resolve",
        "ok" if not broken else "error",
        detail=(
            "every cross-document link and every in-page anchor resolves"
            if not broken else "; ".join(broken[:5])
        ),
        fix="" if not broken else "fix the link, or the heading it points at",
        data={"pages": len(result.pages) - 1, "broken": broken[:20]},
    ))

    remote = [problem for problem in result.problems if problem not in broken]
    findings.append(Finding(
        "docs.network", "the site fetches nothing from the network", "ok" if not remote else "error",
        detail=(
            f"no remote resource in {len(result.pages)} page(s); "
            f"{len(result.external_links)} external hyperlink(s) (clicked, not fetched)"
            if not remote else "; ".join(remote[:5])
        ),
        fix="" if not remote else "inline the resource, or drop it",
        data={"external_links": result.external_links},
    ))

    missing_published = [
        name for name in result.images_missing
        if not any(prefix in name for prefix in ("/out/", "out/"))
    ]
    if result.images_missing:
        severity = "error" if (missing_published and strict_images) else "warning"
        findings.append(Finding(
            "docs.images",
            f"{len(result.images_missing)} figure(s) referenced but not in this tree",
            severity,
            detail=(
                "they are generated by a run (for example `odock surface`), so a clone "
                "does not have them; the site says where each one goes"
            ),
            fix="generate the figures, or remove the reference",
            data={"missing": result.images_missing},
        ))
    else:
        findings.append(Finding(
            "docs.images", f"{result.images_copied} figure(s) copied into the site", "ok",
            detail="every image reference resolves",
        ))

    titles = [
        entry["page"] for entry in result.pages
        if not re.search(r"<h1\b", (Path(result.out) / entry["page"]).read_text(encoding="utf-8"))
    ]
    findings.append(Finding(
        "docs.titles", "every page has a title", "ok" if not titles else "error",
        detail="each page starts with an <h1>" if not titles else ", ".join(titles),
        fix="" if not titles else "give the document a top-level heading",
    ))

    findings.append(Finding(
        "docs.reference",
        f"the API reference covers {result.api_modules} module(s), "
        f"{result.api_names} public name(s)",
        "ok",
        detail=(
            f"read from the source by `ast`, never by importing it. "
            f"{len(result.api_undocumented)} public name(s) have no docstring and are "
            "reported in the reference rather than omitted"
            + (": " + ", ".join(result.api_undocumented[:6]) if result.api_undocumented else "")
        ),
        fix="" if not result.api_undocumented
        else "document the names, or make them private (an underscore) if they are not public",
        data={"modules": result.api_modules, "public_names": result.api_names,
              "undocumented": result.api_undocumented},
    ))

    tutorial_severity = "ok" if result.tutorial_ok else (
        "info" if result.tutorial_skipped else "error"
    )
    findings.append(Finding(
        "docs.tutorial",
        f"the tutorial ran {result.tutorial_steps} step(s) in "
        f"{result.tutorial_seconds:.1f} s" if result.tutorial_ok
        else ("the tutorial was skipped: its bundled inputs are missing"
              if result.tutorial_skipped else "the tutorial did not complete"),
        tutorial_severity,
        detail=(
            "its measured numbers are on the page, because the page is rendered from "
            "the run: " + "; ".join(
                f"{name}: " + ", ".join(f"{key}={value}" for key, value in numbers.items())
                for name, numbers in list(result.tutorial_numbers.items())[:4]
            )
            if result.tutorial_ok
            else (
                "a tree without the bundled demo data cannot run the real toolchain; "
                "`make demo` provides it, and the gate is a gate again"
                if result.tutorial_skipped
                else "see the build problems: a tutorial whose numbers can drift must "
                     "fail the release rather than ship"
            )
        ),
        fix="" if result.tutorial_ok else (
            "run `make demo`" if result.tutorial_skipped
            else "run `odock tutorial` and fix the failing step"
        ),
        data={"steps": result.tutorial_steps, "seconds": result.tutorial_seconds,
              "numbers": result.tutorial_numbers},
    ))

    expected = {source.page for source in source_documents(base)}
    expected |= {"index.html"}
    if result.api_modules:
        expected |= {"api.html"} | {module.page for module in api_modules(base)}
    if all(path.is_file() for path in (base / "tests" / "data" / "3PTB.pdb",
                                       base / "demo" / "libraries" / "library.smi")):
        expected.add(TUTORIAL_PAGE)
    generated = {entry["page"] for entry in result.pages}
    missing, extra = sorted(expected - generated), sorted(generated - expected)
    findings.append(Finding(
        "docs.set",
        f"the site covers all {len(expected) - 1} source(s), including the generated "
        "reference and tutorial",
        "ok" if not missing and not extra else "error",
        detail=(
            "the generated set equals the source set"
            if not missing and not extra
            else f"missing: {missing}; unexpected: {extra}"
        ),
        fix="" if not missing and not extra else "re-run the build; this is a bug in `docs build`",
        data={"expected": len(expected) - 1, "generated": len(generated) - 1},
    ))

    findings.append(Finding(
        "docs.search", f"the search index is {result.search_bytes:,} bytes", "ok",
        detail=(
            f"{result.search_entries} section entry/entries over titles, headings and a "
            "200-character extract per section; matching is case-insensitive substring, "
            "so it does not stem, rank, or search inside code fences"
        ),
        data={"bytes": result.search_bytes, "entries": result.search_entries},
    ))
    for warning in result.warnings:
        findings.append(Finding("docs.note", warning.split(":")[0], "info",
                                detail=warning, data={}))
    return findings


# ---------------------------------------------------------------------------
# Serving
# ---------------------------------------------------------------------------


def serve_site(directory, *, host: str = "127.0.0.1", port: int = 8000,
               quiet: bool = False, ready: Optional[Callable[[Any], None]] = None) -> int:
    """Serve an already-built site over HTTP until interrupted.

    Optional on purpose: the files work when opened directly, and this exists only
    because a local address is convenient for reading on a tablet.  It binds the
    loopback interface by default and serves the directory read-only — it is not a
    way to read a release, and the release gates never use it.

    `ready` is called with the bound server *before* it starts serving.  That is
    what lets a caller print the real port when zero was requested, and what lets a
    test exercise the server for real (fetch a page, then shut it down) instead of
    asserting that the code exists.
    """
    import functools
    import http.server
    import socketserver

    root = Path(directory).resolve()
    if not (root / "index.html").is_file():
        raise DocsError(
            f"{_project.portable_path(root)} has no index.html",
            fix="run `odock docs build` first (or pass the directory it wrote to)",
        )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    with socketserver.TCPServer((host, port), handler) as server:
        server.allow_reuse_address = True
        actual = server.server_address[1]
        if not quiet:
            print(f"serving {_project.portable_path(root)} on http://{host}:{actual}/")
            print("the same files work from disk; press Ctrl-C to stop")
        if ready is not None:
            ready(server)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            if not quiet:
                print("\nstopped")
    return 0


# ---------------------------------------------------------------------------
# The command line
# ---------------------------------------------------------------------------


def _default_root() -> Path:
    from .doctor import _default_root as doctor_root

    found = doctor_root()
    return found if found is not None else Path.cwd()


def _default_site_dir(root) -> Path:
    """``out/docs-site``: a build product belongs under the ignored ``out/``.

    Not ``<root>/docs-site``: a directory in the repository root that ``.gitignore``
    does not know about is content ``release stage`` would happily publish.
    """
    return Path(root) / "out" / "docs-site"


def cmd_docs_build(args) -> int:
    root = Path(args.root) if args.root else _default_root()
    out = Path(args.out) if args.out else _default_site_dir(root)
    try:
        result = build_site(root, out, title=args.title)
    except DocsError as exc:
        print(f"odock docs build: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(result.as_dict(), indent=2, ensure_ascii=False))
    else:
        print(result.text())
    return 0 if result.ok else 1


def cmd_docs_check(args) -> int:
    root = Path(args.root) if args.root else _default_root()
    out = Path(args.out) if args.out else _default_site_dir(root)
    findings = check_site(root, out, strict_images=bool(args.strict_images))
    counts = {"ok": 0, "info": 0, "warning": 0, "error": 0}
    for finding in findings:
        counts[finding.severity] = counts.get(finding.severity, 0) + 1
    if args.json:
        print(json.dumps({
            "format": "odock-docs-check",
            "ok": counts["error"] == 0,
            "counts": counts,
            "findings": [
                {"key": finding.key, "severity": finding.severity, "title": finding.title,
                 "detail": finding.detail, "fix": finding.fix}
                for finding in findings
            ],
        }, indent=2, ensure_ascii=False))
    else:
        for finding in findings:
            mark = {"ok": "ok  ", "info": "note", "warning": "warn", "error": "FAIL"}[
                finding.severity
            ]
            print(f"[{mark}] {finding.title}")
            if finding.detail and finding.severity != "ok":
                print(f"        {finding.detail[:300]}")
            if finding.fix:
                print(f"        fix: {finding.fix}")
    return 1 if counts["error"] else 0


def cmd_docs_serve(args) -> int:
    root = Path(args.root) if args.root else _default_root()
    directory = Path(args.directory) if args.directory else _default_site_dir(root)
    try:
        return serve_site(directory, host=args.host, port=int(args.port))
    except DocsError as exc:
        print(f"odock docs serve: {exc}", file=sys.stderr)
        return 2


def add_docs_parser(sub) -> None:
    """Register ``odock docs build|check|serve`` on `sub`."""
    choices = getattr(sub, "choices", None)
    if isinstance(choices, Mapping) and "docs" in choices:
        return
    parser = sub.add_parser(
        "docs",
        help="render the documentation into a browsable static site",
        description=(
            "Render every published markdown document (docs/, the repository root "
            "and demo/) into one self-contained directory: an index grouped by "
            "audience, a page with a sidebar per document, working anchors, "
            "resolving cross-links, a search index built at generation time, and a "
            "guard that fails on a broken link, a missing image, a page without a "
            "title or anything fetched from the network."
        ),
    )
    actions = parser.add_subparsers(dest="action", required=True)

    build = actions.add_parser("build", help="render the site")
    build.add_argument("-o", "--out", help="the output directory (default: out/docs-site)")
    build.add_argument("--title", default="OpenDocking documentation")
    build.add_argument("--root", help="the repository root (default: this checkout)")
    build.add_argument("--json", action="store_true", help="machine-readable result")
    build.set_defaults(func=cmd_docs_build)

    check = actions.add_parser(
        "check",
        help="build the site and guard it (links, images, titles, network, set)",
        description=(
            "The gate `release check` runs.  Errors: a link or anchor that does not "
            "resolve, an image that is not in the tree, a page without a title, a "
            "generated set that differs from the source set, any remote resource.  "
            "Warnings: figures generated by a run (--strict-images makes those "
            "errors too)."
        ),
    )
    check.add_argument("-o", "--out", help="where to build (default: out/docs-site)")
    check.add_argument("--root", help="the repository root (default: this checkout)")
    check.add_argument("--strict-images", action="store_true",
                       help="a missing figure is an error, not a warning")
    check.add_argument("--json", action="store_true", help="machine-readable result")
    check.set_defaults(func=cmd_docs_check)

    serve = actions.add_parser(
        "serve",
        help="serve an already-built site on the loopback interface",
        description=(
            "Serves the directory read-only on 127.0.0.1.  Convenience only: the "
            "pages load their own stylesheet and search index, so opening the files "
            "directly works, and a release is never read through this server."
        ),
    )
    serve.add_argument("directory", nargs="?", help="the built site (default: out/docs-site)")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--root", help="the repository root (default: this checkout)")
    serve.set_defaults(func=cmd_docs_serve)

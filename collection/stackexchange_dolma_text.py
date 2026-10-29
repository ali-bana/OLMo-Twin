#!/usr/bin/env python3
"""Render a StackExchange post's HTML the way OLMo's training data has it.

Dolma's `stackexchange` component (dolmino-mix-1124, snapshot provenance
20241028) holds ONE record per (question, answer) pair with id
`{forum}-{question_id}-{answer_id}` and text

    title \\n\\n\\n question_body \\n\\n\\n answer_body

where each body is HTML rendered to plain text. The rendering is not any
off-the-shelf library: html2text with every plausible option reached 9/150
byte-identical, bs4/inscriptis/markdownify 0-2/150. The rules below were
reverse-engineered against pairs present in both our archive.org dump and
dolmino whose LastEditDate matches (so the HTML is identical), and verified
byte for byte (`build_stackexchange_pool.py calibrate` measures the agreement).

Rules, each one measured against the corpus:
  * block elements (p, div, pre, lists, blockquote, h*, table) are separated
    by one blank line; an EMPTY <p> still counts as a block
  * newlines inside text nodes collapse to a space; only <br> makes a newline
  * inline <code> is plain text; <pre> blocks are verbatim (own indentation,
    no extra indent, no fences) and keep their trailing newline
  * <ul><li> -> "  • item", <ol><li> -> "  1. item"; items of a tight list
    join with "\\n", items of a loose list (an <li> containing <p>) with a
    blank line; lines after the first inside an <li> -- <br> continuations
    and nested lists -- are indented four spaces
  * <a> -> its anchor text; <img> -> its alt text; emphasis/strong/sup/sub/
    kbd/span/s/del -> plain text; <blockquote> -> its blocks, no "> " prefix
  * the rendered body is LEFT-stripped only. This is why a list that opens a
    body appears as "1. ..." with no indent while the same list elsewhere
    appears as "  1. ...", and why a body ending in <pre> ends in "\\n".
"""

from __future__ import annotations

import re

from bs4 import BeautifulSoup, NavigableString, Tag

BLOCK = {"p", "div", "pre", "ul", "ol", "blockquote", "h1", "h2", "h3", "h4",
         "h5", "h6", "table", "dl", "hr"}
WS_RE = re.compile(r"[ \t\r\n\f]+")
BR = "\x00"  # sentinel for <br>, resolved after whitespace collapsing


def _inline(node, in_pre: bool = False) -> str:
    if isinstance(node, NavigableString):
        if node.parent is not None and node.parent.name in ("script", "style"):
            return ""
        return str(node)
    if not isinstance(node, Tag):
        return ""
    if node.name == "br":
        return BR
    if node.name == "img":
        return node.get("alt", "") or ""
    return "".join(_inline(c, in_pre) for c in node.children)


def _collapse(text: str) -> str:
    """Browser-style whitespace collapse; <br> sentinels become newlines and
    each resulting line is stripped."""
    lines = WS_RE.sub(" ", text).split(BR)
    return "\n".join(ln.strip() for ln in lines).strip()


def _indent_continuation(text: str, n: int = 4) -> str:
    lines = text.split("\n")
    return "\n".join([lines[0]] + [(" " * n + ln) if ln else ln for ln in lines[1:]])


def _indent_all(text: str, n: int) -> str:
    return "\n".join((" " * n + ln) if ln else ln for ln in text.split("\n"))


def _li_body(li: Tag) -> str:
    """Content of one <li>. Measured layout: the item's own text follows the
    marker; <br> continuation lines and further paragraphs are indented four
    spaces; a nested list follows after a single newline and is indented two
    (its own markers carry two more, so it sits at four)."""
    parts: list[tuple[str, bool]] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            txt = _collapse("".join(buf))
            buf.clear()
            if txt:
                parts.append((txt, False))

    for child in li.children:
        if isinstance(child, NavigableString):
            buf.append(str(child))
        elif isinstance(child, Tag) and child.name in ("ul", "ol"):
            flush()
            parts.append((_render_list(child, ordered=(child.name == "ol")), True))
        elif isinstance(child, Tag) and child.name in BLOCK:
            flush()
            parts.append((_blocks(child) if child.find(list(BLOCK)) else _collapse(_inline(child)), False))
        elif isinstance(child, Tag):
            buf.append(_inline(child))
    flush()
    out = ""
    for i, (txt, is_list) in enumerate(parts):
        if is_list:
            out += "\n" + _indent_all(txt, 2)
        elif i == 0:
            out = _indent_continuation(txt, 4)
        else:
            out += "\n\n" + _indent_all(txt, 4)
    return out


def _render_list(tag: Tag, ordered: bool) -> str:
    items, n, loose = [], 0, False
    for li in tag.find_all("li", recursive=False):
        n += 1
        if li.find("p") is not None:      # a nested list alone does not make it loose
            loose = True
        marker = f"  {n}. " if ordered else "  • "
        items.append(marker + _li_body(li))
    return ("\n\n" if loose else "\n").join(items)


def _blocks(container: Tag) -> str:
    """Render a container's children as blank-line-separated blocks."""
    blocks: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if buf:
            txt = _collapse("".join(buf))
            buf.clear()
            if txt:
                blocks.append(txt)

    for child in container.children:
        if isinstance(child, NavigableString):
            buf.append(str(child))
            continue
        if not isinstance(child, Tag):
            continue
        name = child.name
        if name not in BLOCK:
            buf.append(_inline(child))
            continue
        flush()
        if name == "pre":
            code = child.find("code")
            txt = _inline(code if code is not None else child, in_pre=True).replace(BR, "\n")
            blocks.append(txt.rstrip("\n"))
        elif name in ("ul", "ol"):
            blocks.append(_render_list(child, ordered=(name == "ol")))
        elif name == "blockquote":
            inner = _blocks(child)
            blocks.append(inner)
        elif name == "hr":
            continue
        elif name == "table":
            rows = []
            for tr in child.find_all("tr"):
                cells = [_collapse(_inline(td)) for td in tr.find_all(["td", "th"])]
                rows.append(" ".join(c for c in cells if c))
            blocks.append("\n".join(r for r in rows if r))
        else:  # p, div, h*, dl
            if child.find(list(BLOCK)) is not None:
                blocks.append(_blocks(child))
            else:
                # an empty <p> is still a block (measured: produces a blank line)
                blocks.append(_collapse(_inline(child)))
    flush()
    return "\n\n".join(blocks)


def html_to_text(html: str) -> str:
    """Dolma's rendering of one post body: blocks joined by blank lines, then
    LEFT-stripped (trailing whitespace such as a <pre>'s final newline is kept)."""
    soup = BeautifulSoup(html, "html.parser")
    return _blocks(soup).lstrip()


def record_text(title: str, question_html: str, answer_html: str) -> str:
    """The exact text Dolma stores for one (question, answer) pair."""
    return title.strip() + "\n\n\n" + html_to_text(question_html) + "\n\n\n" + html_to_text(answer_html)

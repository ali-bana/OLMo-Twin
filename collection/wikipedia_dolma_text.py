#!/usr/bin/env python3
"""Turn Wikipedia page XML into the text OLMo's training data holds for it.

Dolma's `wiki` component (the same bytes in olmo-mix-1124 and dolmino-mix-1124,
snapshot 2023-04-02) was produced with WikiExtractor. This was established
empirically, not from documentation: for pages whose last revision predates
the snapshot -- so the wikitext in our 2026 dump is identical to what Dolma
processed -- WikiExtractor 3.0.8 with `--no-templates` reproduces Dolma's
record **byte for byte on 37 of 40 pages**; the remaining two differ locally
(similarity 0.947 and 0.9994) and one page is not emitted. `mwparserfromhell`
(used previously) does not reproduce the format at all: Dolma's text renders
section headings as `History.` with a trailing period, italics as "quoted"
text, and drops templates leaving gaps ("She is in length and has a cargo
capacity of .") -- all WikiExtractor signatures.

Record text = title + "\n\n" + body. Templates are NOT expanded (Dolma did
not; identical output with and without --no-templates on unchanged pages).

Version matters: 3.0.6 drops pages 3.0.8 keeps and needs a regex patch on
Python 3.11+; 3.1.0 collapses whitespace differently (`measured.` vs Dolma's
`measured .`). The pin is enforced at import.
"""

from __future__ import annotations

import bz2
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REQUIRED_WIKIEXTRACTOR = "3.0.8"
PAGE_RE = re.compile(r"  <page>\n.*?  </page>\n", re.S)
PAGE_ID_RE = re.compile(r"\n    <id>(\d+)</id>\n")

# Appended to every mini-dump so WikiExtractor's lost-last-article bug eats
# this page instead of a real one; filtered out of every result.
SENTINEL_ID = "999999999999"
SENTINEL_PAGE = (
    "  <page>\n    <title>WikiExtractorFlushSentinel</title>\n    <ns>0</ns>\n"
    f"    <id>{SENTINEL_ID}</id>\n    <revision>\n      <id>1</id>\n"
    "      <timestamp>2000-01-01T00:00:00Z</timestamp>\n      <format>text/x-wiki</format>\n"
    "      <text xml:space=\"preserve\">Sentinel page absorbing the final-article flush loss "
    "in wikiextractor 3.0.8. It is never part of any result.</text>\n    </revision>\n  </page>\n"
)


def check_wikiextractor() -> str:
    from importlib.metadata import version, PackageNotFoundError
    try:
        v = version("wikiextractor")
    except PackageNotFoundError:
        raise SystemExit("wikiextractor is not installed; pip install wikiextractor==3.0.8")
    if v != REQUIRED_WIKIEXTRACTOR:
        raise SystemExit(
            f"wikiextractor {v} installed but {REQUIRED_WIKIEXTRACTOR} is required. The "
            "fidelity measurement (37/40 byte-identical to Dolma) holds for that version "
            "only; 3.0.6 drops pages and 3.1.0 changes whitespace."
        )
    return v


def dump_header(dump_dir: Path) -> str:
    """The <mediawiki ...><siteinfo>...</siteinfo> prologue from the real dump.

    WikiExtractor reads namespaces and the base URL from siteinfo, and parses
    the file line by line, so a mini-dump must reuse the genuine, line-formatted
    header rather than a hand-written one (a one-line header yields zero
    articles with no error).
    """
    data = sorted(p for p in dump_dir.glob("*pages-articles-multistream1.xml*.bz2")
                  if "index" not in p.name)
    if not data:
        raise FileNotFoundError(f"no multistream1 data file under {dump_dir}")
    with open(data[0], "rb") as fh:
        head = bz2.BZ2Decompressor().decompress(fh.read(400_000)).decode("utf-8", "replace")
    end = head.find("</siteinfo>")
    if end < 0:
        raise ValueError("siteinfo not found in the first stream of the dump")
    return head[: end + len("</siteinfo>")] + "\n"


def page_xml_blocks(blob: str) -> dict[str, str]:
    """{page_id: '  <page>...</page>\\n'} for every page in a decompressed stream."""
    out: dict[str, str] = {}
    for m in PAGE_RE.finditer(blob):
        block = m.group(0)
        pid = PAGE_ID_RE.search(block)
        if pid:
            out[pid.group(1)] = block
    return out


def extract_texts(page_blocks: list[str], header: str, processes: int = 1) -> dict[str, str]:
    """Run WikiExtractor over the given page XML blocks; return {page_id: text}.

    Pages WikiExtractor does not emit (redirects, pages it drops) are absent
    from the result rather than present with empty text.
    """
    check_wikiextractor()
    if not page_blocks:
        return {}
    with tempfile.TemporaryDirectory(prefix="wx_") as td:
        dump = Path(td) / "dump.xml"
        with dump.open("w", encoding="utf-8") as fh:
            fh.write(header)
            for b in page_blocks:
                fh.write(b)
            # WikiExtractor 3.0.8 never flushes its LAST article: N pages in,
            # N-1 out (1 page -> a 0-byte file, while the log still says
            # "extraction of 1 articles"). A sentinel page absorbs the loss and
            # is dropped below. Measured: without it, exactly one page was
            # "not extracted" in every batch -- always the final one.
            fh.write(SENTINEL_PAGE)
            fh.write("</mediawiki>\n")
        out = Path(td) / "out"
        r = subprocess.run(
            [sys.executable, "-m", "wikiextractor.WikiExtractor", str(dump),
             "-o", str(out), "--json", "--no-templates", "--processes", str(processes)],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError(f"wikiextractor failed: {r.stderr[-800:]}")
        texts: dict[str, str] = {}
        for f in sorted(out.rglob("wiki_*")):
            with f.open(encoding="utf-8") as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    if str(rec["id"]) == SENTINEL_ID:
                        continue
                    texts[str(rec["id"])] = rec["title"] + "\n\n" + rec["text"]
        return texts


if __name__ == "__main__":  # tiny self-check on a dump
    import argparse
    ap = argparse.ArgumentParser(description="print Dolma-style text for page ids")
    ap.add_argument("--dump-dir", type=Path, required=True)
    ap.add_argument("--blob-file", type=Path, help="decompressed multistream chunk to read pages from")
    ap.add_argument("ids", nargs="+")
    a = ap.parse_args()
    hdr = dump_header(a.dump_dir)
    blocks = page_xml_blocks(a.blob_file.read_text(encoding="utf-8", errors="replace"))
    got = extract_texts([blocks[i] for i in a.ids if i in blocks], hdr)
    for i in a.ids:
        print(f"=== {i} ===\n{got.get(i, '<not emitted>')[:600]}\n")

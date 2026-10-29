#!/usr/bin/env python3
"""Build Wikipedia candidate pools for the OLMo 2 injection set.

Two groups, because they answer different questions:

  A  created   -- page's FIRST revision is after the model release date. The
                  whole article postdates every corpus snapshot, so it is a
                  non-member by construction.
  B  edited    -- page existed before the release date but has been edited
                  since. Only part of the text is new, so these are expected to
                  overlap training data and are the harder, more interesting
                  case. Kept separate; never pooled with A.

Why these are safe by date: the corpora's `wiki` component is a 2023-04-25
snapshot and `dclm` is Common Crawl through 2022-12. An article created after
the release date (2025-04-17) cannot be in either. The scan still runs, because
a new page can quote old text verbatim -- that is exactly what group B
measures and what the verbatim-span exclusion rule of Stage G (about 50
tokens) is meant to catch.

Text format. Pages are rendered with WikiExtractor 3.0.8 (--no-templates),
title + blank line + body -- the recipe measured to reproduce Dolma's own
wiki records byte for byte on 37 of 40 pages unchanged since the 2023
snapshot (see wikipedia_dolma_text.py). Both the Dolma-format text (.txt) and
its normalized form (.zst, what the scan fingerprints) are stored.

Sampling is a pure function of --seed: page_meta is written sorted by page
id and every population is sorted before drawing. Group labels from this
stage are PROVISIONAL (creation-log titles); verify_wikipedia_creation_dates
re-labels from each page's true first revision.

Stages
    scan-dump   one pass over pages-articles: id, title, last-revision date
    sample      pick per month per group, extract text, normalize, compress
"""

from __future__ import annotations

import argparse
import bz2
import csv
import gzip
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import zstandard as zstd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import olmo2_midtrain_membership as mm  # noqa: E402
import wikipedia_dolma_text as wdt  # noqa: E402  WikiExtractor 3.0.8 wrapper, the verified Dolma recipe

RELEASE_DATE = "2025-04-17"
MIN_WORDS = 200          # the cohort spec excludes stubs
PAGE_RE = re.compile(r"<page>(.*?)</page>", re.S)
TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S)
ID_RE = re.compile(r"<id>(\d+)</id>")
TS_RE = re.compile(r"<timestamp>([^<]+)</timestamp>")
TEXT_RE = re.compile(r"<text[^>]*>(.*?)</text>", re.S)
NS_RE = re.compile(r"<ns>(\d+)</ns>")
REDIRECT_RE = re.compile(r"<redirect[ />]")
UNESC = [("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#039;", "'"), ("&amp;", "&")]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def unescape(s: str) -> str:
    for a, b in UNESC:
        s = s.replace(a, b)
    return s


# --------------------------------------------------------------------------
# stage 1: metadata pass over the article dump
# --------------------------------------------------------------------------


def _scan_one(path: str) -> list[tuple[str, str, str]]:
    """(page_id, title, last_revision_timestamp) for main-namespace articles."""
    out = []
    try:
        with bz2.open(path, "rt", encoding="utf-8", errors="replace") as fh:
            buf = []
            for line in fh:
                buf.append(line)
                if "</page>" not in line:
                    continue
                chunk = "".join(buf)
                buf = []
                if NS_RE.search(chunk) and NS_RE.search(chunk).group(1) != "0":
                    continue
                if REDIRECT_RE.search(chunk):
                    continue
                t = TITLE_RE.search(chunk)
                i = ID_RE.search(chunk)
                ts = TS_RE.search(chunk)
                if t and i and ts:
                    out.append((i.group(1), unescape(t.group(1)), ts.group(1)))
    except Exception as exc:
        log(f"  shard failed {os.path.basename(path)}: {exc}")
    return out


def cmd_scan_dump(args: argparse.Namespace) -> int:
    shards = sorted(
        str(p) for p in args.dump_dir.glob("*pages-articles-multistream*.bz2")
        if "index" not in p.name
    )
    log(f"scanning {len(shards)} article shards for page metadata")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "page_meta.tsv.gz"
    collected: list[tuple[str, str, str]] = []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, rows in enumerate(pool.imap_unordered(_scan_one, shards), 1):
            collected.extend(rows)
            log(f"  shard {i}/{len(shards)}  pages so far {len(collected):,}")
    # Sorted by page id so the file -- and everything sampled from it under a
    # fixed seed -- is reproducible. imap_unordered returns shards in worker
    # completion order, which differs run to run.
    collected.sort(key=lambda r: int(r[0]))
    with gzip.open(out_path, "wt") as out:
        out.write("page_id\ttitle\tlast_revision\n")
        for pid, title, ts in collected:
            out.write(f"{pid}\t{title}\t{ts}\n")
    log(f"wrote {out_path} ({len(collected):,} main-namespace articles, sorted by page id)")
    return 0


# --------------------------------------------------------------------------
# stage 2: sample and extract text
# --------------------------------------------------------------------------


def load_index(dump_dir: Path, wanted: set[str]) -> dict[str, tuple[str, int, int]]:
    """title -> (data file, stream offset, page id), from the multistream index."""
    loc: dict[str, tuple[str, int, int]] = {}
    idx = sorted(p for p in dump_dir.glob("*index*.bz2"))
    pat = re.compile(r"(.*)-index(\d+)\.txt-(p\d+p\d+)\.bz2$")
    for n, p in enumerate(idx, 1):
        m = pat.match(p.name)
        if not m:
            continue
        data_name = f"{m.group(1)}-multistream{m.group(2)}.xml-{m.group(3)}.bz2"
        # index file is named ...-pages-articles-multistream-index<N>...
        data_name = data_name.replace("-multistream-multistream", "-multistream")
        if not (dump_dir / data_name).exists():
            continue
        with bz2.open(p, "rt", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.rstrip("\n").split(":", 2)
                if len(parts) < 3:
                    continue
                if parts[2] in wanted:
                    loc[parts[2]] = (data_name, int(parts[0]), int(parts[1]))
        if n % 20 == 0:
            log(f"  index {n}/{len(idx)}: resolved {len(loc):,}/{len(wanted):,}")
    return loc


def _extract_group(task):
    """Return the raw `<page>...</page>` XML blocks for the wanted titles.

    The blocks are handed to WikiExtractor verbatim (see wikipedia_dolma_text),
    which is what reproduces Dolma's text byte for byte. The previous version
    pulled the wikitext out and cleaned it with mwparserfromhell, whose output
    does not match the corpus (no `Heading.` periods, no "quoted" italics, no
    template gaps).
    """
    dump_dir, data_name, offset, titles = task
    out = []
    try:
        with open(os.path.join(dump_dir, data_name), "rb") as fh:
            fh.seek(offset)
            blob = bz2.BZ2Decompressor().decompress(fh.read(8_000_000))
    except Exception:
        return out
    xml = blob.decode("utf-8", "replace")
    want = set(titles)
    for pm in wdt.PAGE_RE.finditer(xml):
        block = pm.group(0)
        tm = TITLE_RE.search(block)
        if not tm:
            continue
        title = unescape(tm.group(1))
        if title not in want or REDIRECT_RE.search(block):
            continue
        out.append((title, block))
    return out


def cmd_sample(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log(f"wikiextractor {wdt.check_wikiextractor()} (pinned; see wikipedia_dolma_text)")
    header = wdt.dump_header(args.dump_dir)
    created: dict[str, str] = {}
    with gzip.open(args.creations, "rt") as fh:
        next(fh)
        for line in fh:
            ts, title = line.rstrip("\n").split("\t", 1)
            created[title] = ts

    log(f"pages created in window: {len(created):,}")
    bucket_a: dict[str, list[str]] = defaultdict(list)
    bucket_b: dict[str, list[str]] = defaultdict(list)
    with gzip.open(args.page_meta, "rt") as fh:
        next(fh)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            pid, title, last = parts
            if title in created:
                if created[title][:10] >= RELEASE_DATE:
                    bucket_a[created[title][:7]].append(title)
            elif last[:10] >= RELEASE_DATE:
                # existed before the release date, edited after it
                bucket_b[last[:7]].append(title)
    log(f"group A (created after {RELEASE_DATE}): {sum(len(v) for v in bucket_a.values()):,}")
    log(f"group B (pre-existing, edited after):   {sum(len(v) for v in bucket_b.values()):,}")

    rng = random.Random(args.seed)
    wanted: dict[str, tuple[str, str]] = {}
    for grp, bucket in (("A", bucket_a), ("B", bucket_b)):
        for month in sorted(bucket):
            # sorted so the draw is a pure function of the seed
            pop = sorted(bucket[month])
            take = min(args.per_month, len(pop))
            for t in rng.sample(pop, take):
                wanted[t] = (grp, month)
            log(f"  {grp} {month}: population {len(pop):,} -> sampled {take:,}")
    log(f"total titles to extract: {len(wanted):,}")

    loc = load_index(args.dump_dir, set(wanted))
    log(f"resolved {len(loc):,}/{len(wanted):,} titles in the multistream index")

    groups = defaultdict(list)
    for t, (fn, off, pid) in loc.items():
        groups[(fn, off)].append(t)
    tasks = [(str(args.dump_dir), fn, off, ts) for (fn, off), ts in groups.items()]
    log(f"extracting from {len(tasks):,} multistream blocks")

    texts_dir = args.out_dir / "texts"
    texts_dir.mkdir(parents=True, exist_ok=True)
    cctx = zstd.ZstdCompressor(level=6)
    rows = []
    short = not_emitted = 0
    pid2title = {str(v[2]): k for k, v in loc.items()}
    batch: list[str] = []

    def flush(batch: list[str]) -> None:
        nonlocal short, not_emitted
        # WikiExtractor over a mini-dump of these pages; pages it drops are
        # simply absent (redirects, and the ~2.5% it does not emit).
        texts = wdt.extract_texts(batch, header, processes=1)
        emitted = set()
        for pid, text in texts.items():
            emitted.add(pid)
            title = pid2title.get(pid)
            if title is None:
                continue
            grp, month = wanted[title]
            nw = len(text.split())
            if nw < MIN_WORDS:
                short += 1
                continue
            norm = mm.normalize(text)
            if len(norm) < 400:
                short += 1
                continue
            path = texts_dir / f"{grp}_{pid}.zst"
            path.write_bytes(cctx.compress(norm))
            raw_path = texts_dir / f"{grp}_{pid}.txt"
            raw_path.write_text(text, encoding="utf-8")  # Dolma-format text, for review
            rows.append({
                "doc_id": f"wiki{grp}:{pid}", "group": grp, "month": month,
                "title": title, "n_words": nw, "norm_bytes": len(norm),
                "text_path": str(path), "raw_text_path": str(raw_path),
                "parser_status": "ok",
            })
        not_emitted += sum(1 for b in batch if (m := wdt.PAGE_ID_RE.search(b)) and m.group(1) not in emitted)

    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, res in enumerate(pool.imap_unordered(_extract_group, tasks, chunksize=4), 1):
            for _title, block in res:
                batch.append(block)
                if len(batch) >= args.batch_pages:
                    flush(batch)
                    batch = []
            if i % 200 == 0:
                log(f"  block {i:,}/{len(tasks):,}  kept {len(rows):,}  short {short:,}  not emitted {not_emitted:,}")
    if batch:
        flush(batch)

    out = args.out_dir / "wiki_pools.csv.gz"
    cols = ["doc_id", "group", "month", "title", "n_words", "norm_bytes",
            "text_path", "raw_text_path", "parser_status"]
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    per = Counter((r["group"], r["month"]) for r in rows)
    for grp in ("A", "B"):
        months = sorted(m for g, m in per if g == grp)
        log(f"group {grp}: {sum(per[(grp,m)] for m in months):,} articles")
        for m in months:
            log(f"   {m}: {per[(grp,m)]:,}")
    log(f"dropped as too short: {short:,}   not emitted by WikiExtractor: {not_emitted:,}")
    log(f"wrote {out}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("scan-dump", help="page id/title/last-revision for every article")
    s.add_argument("--dump-dir", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    s.set_defaults(func=cmd_scan_dump)

    q = sub.add_parser("sample", help="sample both groups and extract clean text")
    q.add_argument("--dump-dir", type=Path, required=True)
    q.add_argument("--creations", type=Path, required=True)
    q.add_argument("--page-meta", type=Path, required=True)
    q.add_argument("--out-dir", type=Path, required=True)
    q.add_argument("--per-month", type=int, default=1200)
    q.add_argument("--seed", type=int, default=20260915)
    q.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    q.add_argument("--batch-pages", type=int, default=500,
                   help="pages per WikiExtractor invocation")
    q.set_defaults(func=cmd_sample)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

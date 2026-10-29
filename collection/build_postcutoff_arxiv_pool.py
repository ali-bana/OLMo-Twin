#!/usr/bin/env python3
"""Build the post-cutoff arXiv candidate pool for the OLMo 2 injection set.

Papers submitted after the model's release date cannot be in its training data,
so they are the trustworthy non-member source (the pre-cutoff pool collapsed
from 27,409 to 1,464 once stage-1 was searched).

Every paper is kept in BOTH representations, because the two training corpora
ingest arXiv through different pipelines and a paper can hide from one while
being plainly present in the other:

  * LaTeX source  -- matches olmo-mix's `arxiv` component (RedPajama/Proof-Pile-2)
  * GROBID PDF    -- matches `pes2o` (S2ORC), which is parsed from the published PDF

Both are normalized with the same function the corpus scan uses, then zstd
compressed, so the fingerprint stage can consume them directly.

Subcommands
    sample   choose IDs per month, fetch metadata, write the manifest
    parse    run the LaTeX cleaner and the peS2o-style PDF parser over them
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time
import urllib.parse

import requests
from collections import Counter, defaultdict
from pathlib import Path

import zstandard as zstd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import olmo2_midtrain_membership as mm  # noqa: E402  (normalize, month helpers)
import redpajama_arxiv_cleaner_standalone as rp  # noqa: E402
import pes2o_style_pdf_parser as pp  # noqa: E402

# OLMo-2-0425-1B was published to the Hub on this date; anything submitted to
# arXiv after it postdates every corpus snapshot used to train the model.
RELEASE_DATE = "2025-04-17"
ARXIV_API = "https://export.arxiv.org/api/query"
USER_AGENT = os.environ.get("OLMO_TWIN_USER_AGENT", "OLMo-Twin/1.0 (research code)")
PDF_SUFFIX_RE = re.compile(r"(v\d+)?\.pdf$")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------
# population
# --------------------------------------------------------------------------


def month_population(arxiv_root: Path, month: str) -> list[str]:
    """IDs that have BOTH a LaTeX source archive and a PDF.

    `src/<mm>` holds `<id>.gz` for LaTeX submissions and `<id>.pdf` for
    PDF-only ones (no LaTeX exists for those, so they cannot be dual-parsed).
    `pdf/<mm>` names files `<id>v1.pdf` for earlier months and `<id>.pdf` for
    later ones -- both spellings must be stripped or the intersection silently
    collapses to a few percent.
    """
    src_dir = arxiv_root / "src" / month
    pdf_dir = arxiv_root / "pdf" / month
    if not src_dir.is_dir() or not pdf_dir.is_dir():
        return []
    latex = {f[:-3] for f in os.listdir(src_dir) if f.endswith(".gz")}
    pdfs = {PDF_SUFFIX_RE.sub("", f) for f in os.listdir(pdf_dir) if f.endswith(".pdf")}
    return sorted(latex & pdfs)


# --------------------------------------------------------------------------
# metadata
# --------------------------------------------------------------------------


def fetch_metadata(ids: list[str], batch: int = 100, delay: float = 3.0) -> dict[str, dict]:
    """arXiv API metadata. Delay honours their request of ~1 call per 3 s."""
    out: dict[str, dict] = {}
    for start in range(0, len(ids), batch):
        chunk = ids[start : start + batch]
        # NOTE: urllib is not usable here. Its plain HTTP/1.1 request is
        # answered with "406 Not Acceptable" no matter which User-Agent or
        # Accept header is set; requests (urllib3) negotiates a connection the
        # API accepts. Verified empirically against six header combinations.
        xml = ""
        for attempt in range(5):
            try:
                resp = requests.get(
                    ARXIV_API,
                    params={"id_list": ",".join(chunk), "max_results": len(chunk)},
                    headers={"User-Agent": USER_AGENT},
                    timeout=120,
                )
                resp.raise_for_status()
                xml = resp.text
                break
            except Exception as exc:
                if attempt == 4:
                    raise SystemExit(
                        f"arXiv metadata request failed after 5 attempts: {exc}\n"
                        "  Refusing to continue -- silently empty metadata would "
                        "drop every paper and look like a legitimate empty pool."
                    )
                time.sleep(5 * (attempt + 1))
        for m in re.finditer(r"<entry>(.*?)</entry>", xml, re.S):
            e = m.group(1)
            gid = re.search(r"<id>([^<]+)</id>", e)
            if not gid:
                continue
            base = gid.group(1).rstrip("/").split("/")[-1].split("v")[0]
            pub = re.search(r"<published>([^<]+)</published>", e)
            ttl = re.search(r"<title>(.*?)</title>", e, re.S)
            summ = re.search(r"<summary>(.*?)</summary>", e, re.S)
            cat = re.search(r'<arxiv:primary_category[^>]*term="([^"]+)"', e)
            out[base] = {
                "published": (pub.group(1)[:10] if pub else ""),
                "title": " ".join(ttl.group(1).split()) if ttl else "",
                "abstract": " ".join(summ.group(1).split()) if summ else "",
                "primary_category": cat.group(1) if cat else "",
            }
        if start + batch < len(ids):
            time.sleep(delay)
        if (start // batch) % 10 == 0:
            log(f"  metadata {min(start + batch, len(ids)):,}/{len(ids):,}")
    return out


CATEGORY_GROUPS = {
    "cs": "cs", "math": "math", "stat": "stat_eess", "eess": "stat_eess",
}
PHYSICS_PREFIXES = (
    "physics", "astro-ph", "cond-mat", "gr-qc", "hep-", "nucl-", "quant-ph",
)


def category_group(primary: str) -> str:
    if not primary:
        return "other"
    head = primary.split(".")[0]
    if head in CATEGORY_GROUPS:
        return CATEGORY_GROUPS[head]
    if primary.startswith(PHYSICS_PREFIXES):
        return "physics"
    return "other"


def cmd_sample(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    picked: list[tuple[str, str]] = []
    for month in args.months:
        pop = month_population(args.arxiv_root, month)
        take = min(args.per_month, len(pop))
        for doc in rng.sample(pop, take):
            picked.append((month, doc))
        log(f"{month}: population {len(pop):,} -> sampled {take:,}")
    log(f"total sampled: {len(picked):,}; fetching metadata")

    meta = fetch_metadata([d for _, d in picked], delay=args.api_delay)
    log(f"metadata returned for {len(meta):,}/{len(picked):,}")

    rows, dropped_date, no_meta = [], 0, 0
    for month, doc in picked:
        m = meta.get(doc)
        if not m:
            no_meta += 1
            continue
        # A 2504 paper submitted before the release date could predate the
        # corpus snapshot, so it is not a safe non-member.
        if m["published"] and m["published"] < RELEASE_DATE:
            dropped_date += 1
            continue
        rows.append({
            "doc_id": doc,
            "yymm": month,
            "month": f"20{month[:2]}-{month[2:]}",
            "published": m["published"],
            "primary_category": m["primary_category"],
            "category_group": category_group(m["primary_category"]),
            "title": m["title"],
            "abstract": m["abstract"],
            "src_path": str(args.arxiv_root / "src" / month / f"{doc}.gz"),
            "pdf_path": "",
            "latex_text_path": "",
            "pdf_text_path": "",
            "latex_status": "",
            "pdf_status": "",
        })
    for r in rows:
        month, doc = r["yymm"], r["doc_id"]
        pdf_dir = args.arxiv_root / "pdf" / month
        for cand in (f"{doc}v1.pdf", f"{doc}.pdf"):
            p = pdf_dir / cand
            if p.exists():
                r["pdf_path"] = str(p)
                break
        if not r["pdf_path"]:
            hits = sorted(pdf_dir.glob(f"{doc}v*.pdf"))
            if hits:
                r["pdf_path"] = str(hits[0])

    log(f"kept {len(rows):,}  (dropped {dropped_date:,} submitted before {RELEASE_DATE},"
        f" {no_meta:,} without metadata)")
    out = args.out_dir / "pool.csv.gz"
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow(r)
    by_month = Counter(r["month"] for r in rows)
    for m in sorted(by_month):
        log(f"   {m}: {by_month[m]:,}")
    log(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------
# parsing
# --------------------------------------------------------------------------


def _latex_one(task: tuple[str, str, str]) -> dict:
    doc_id, src_path, out_path = task
    try:
        text, status = rp.clean_arxiv_source_file(Path(src_path))
    except Exception as exc:
        return {"doc_id": doc_id, "latex_status": f"error_{type(exc).__name__}"}
    if not text:
        return {"doc_id": doc_id, "latex_status": status}
    data = mm.normalize(text)
    if not data:
        return {"doc_id": doc_id, "latex_status": "empty_after_normalize"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(zstd.ZstdCompressor(level=6).compress(data))
    return {"doc_id": doc_id, "latex_status": "ok", "latex_text_path": out_path,
            "latex_bytes": len(data)}


_LM = None


def _pdf_init(lm_path: str | None) -> None:
    # The parser stores the LM as {"lm": {word: logprob}, "floor": float};
    # it is loaded per worker so the dict is shared copy-on-write after fork.
    global _LM
    _LM = json.loads(Path(lm_path).read_text()) if lm_path else None


def _pdf_one(task: tuple[str, str, str, str, str]) -> dict:
    doc_id, pdf_path, out_path, server, meta_abstract = task
    global _LM
    try:
        tei = pp.grobid_fulltext(Path(pdf_path), server)
    except Exception as exc:
        return {"doc_id": doc_id, "pdf_status": f"grobid_{type(exc).__name__}"}
    if not tei:
        return {"doc_id": doc_id, "pdf_status": "grobid_empty"}
    try:
        # The arXiv metadata abstract stands in when GROBID's is missing or is
        # really a figure caption -- see pes2o_style_pdf_parser.choose_abstract.
        doc = pp.tei_to_pes2o_text(tei, _LM, fallback_abstract=meta_abstract)
    except Exception as exc:
        return {"doc_id": doc_id, "pdf_status": f"tei_{type(exc).__name__}"}
    text = (doc or {}).get("text", "")
    if not text:
        return {"doc_id": doc_id, "pdf_status": "tei_empty"}
    data = mm.normalize(text)
    if not data:
        return {"doc_id": doc_id, "pdf_status": "empty_after_normalize"}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_bytes(zstd.ZstdCompressor(level=6).compress(data))
    return {"doc_id": doc_id, "pdf_status": "ok", "pdf_text_path": out_path,
            "pdf_bytes": len(data), "pdf_words": doc.get("n_words", 0),
            "abstract_source": doc.get("abstract_source", "grobid")}


def cmd_parse(args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(gzip.open(args.pool, "rt", newline="")))
    log(f"pool: {len(rows):,} papers")
    tex_dir = args.out_dir / "latex"
    pdf_dir = args.out_dir / "pdf"

    # ---- LaTeX -----------------------------------------------------------
    tex_tasks = []
    for r in rows:
        dest = tex_dir / r["yymm"] / f"{r['doc_id']}.zst"
        if dest.exists() and not args.force:
            r["latex_status"], r["latex_text_path"] = "ok", str(dest)
            continue
        tex_tasks.append((r["doc_id"], r["src_path"], str(dest)))
    log(f"LaTeX: {len(tex_tasks):,} to parse ({len(rows)-len(tex_tasks):,} cached)")
    results: dict[str, dict] = {}
    if tex_tasks:
        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_latex_one, tex_tasks, chunksize=8), 1):
                results[res["doc_id"]] = res
                if i % 500 == 0:
                    ok = sum(1 for v in results.values() if v.get("latex_status") == "ok")
                    log(f"  latex {i:,}/{len(tex_tasks):,}  ok={ok:,}")
    for r in rows:
        if r["doc_id"] in results:
            r.update({k: v for k, v in results[r["doc_id"]].items() if k != "doc_id"})

    # ---- PDF via GROBID --------------------------------------------------
    pdf_tasks = []
    for r in rows:
        if not r.get("pdf_path"):
            r["pdf_status"] = "no_pdf"
            continue
        dest = pdf_dir / r["yymm"] / f"{r['doc_id']}.zst"
        if dest.exists() and not args.force:
            r["pdf_status"], r["pdf_text_path"] = "ok", str(dest)
            continue
        pdf_tasks.append((r["doc_id"], r["pdf_path"], str(dest), args.server,
                          r.get("abstract", "")))
    log(f"PDF: {len(pdf_tasks):,} to parse ({sum(1 for r in rows if r.get('pdf_status')=='ok'):,} cached)")
    presults: dict[str, dict] = {}
    if pdf_tasks:
        ctx = mp.get_context("fork")
        with ctx.Pool(args.grobid_workers, initializer=_pdf_init,
                      initargs=(str(args.unigram_lm) if args.unigram_lm else None,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_pdf_one, pdf_tasks, chunksize=1), 1):
                presults[res["doc_id"]] = res
                if i % 200 == 0:
                    ok = sum(1 for v in presults.values() if v.get("pdf_status") == "ok")
                    log(f"  pdf {i:,}/{len(pdf_tasks):,}  ok={ok:,}")
    for r in rows:
        if r["doc_id"] in presults:
            r.update({k: v for k, v in presults[r["doc_id"]].items() if k != "doc_id"})

    cols = list(rows[0].keys())
    for extra in ("latex_bytes", "pdf_bytes", "pdf_words", "abstract_source"):
        if extra not in cols:
            cols.append(extra)
    out = args.out_dir / "pool_parsed.csv.gz"
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)

    lat = Counter(r.get("latex_status") or "missing" for r in rows)
    pdf = Counter(r.get("pdf_status") or "missing" for r in rows)
    both = sum(1 for r in rows if r.get("latex_status") == "ok" and r.get("pdf_status") == "ok")
    log(f"\nLaTeX status: {dict(lat.most_common(6))}")
    log(f"PDF status:   {dict(pdf.most_common(6))}")
    log(f"BOTH representations ok: {both:,}/{len(rows):,}")
    log(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------
# raw (un-normalized) text -- what is injected and scored
# --------------------------------------------------------------------------
#
# `parse` stores NORMALIZED text (lowercased, punctuation folded), which is
# what the corpus scan fingerprints. Injection and log-probability scoring
# need the text as a model would see it. This pass re-runs both parsers,
# writes the raw text, and VERIFIES that normalize(raw) is byte-identical to
# the stored .zst -- so the scanned representation and the injected one are
# provably the same document. On a mismatch the .zst is never rewritten
# (that would silently desynchronize it from fingerprints already built);
# the paper is flagged and excluded downstream instead.


def _verify_and_write_raw(text: str, zst_path: str, out_txt: str) -> tuple[str, int]:
    stored = zstd.ZstdDecompressor().decompress(Path(zst_path).read_bytes())
    if mm.normalize(text) != stored:
        return "raw_mismatch", 0
    Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    Path(out_txt).write_text(text, encoding="utf-8")
    return "ok", len(text.encode("utf-8"))


def _raw_latex_one(task: tuple[str, str, str, str]) -> dict:
    doc_id, src_path, zst_path, out_txt = task
    try:
        text, status = rp.clean_arxiv_source_file(Path(src_path))
    except Exception as exc:
        return {"doc_id": doc_id, "latex_raw_status": f"error_{type(exc).__name__}"}
    if not text:
        return {"doc_id": doc_id, "latex_raw_status": status}
    st, nbytes = _verify_and_write_raw(text, zst_path, out_txt)
    return {"doc_id": doc_id, "latex_raw_status": st,
            "latex_raw_path": out_txt if st == "ok" else "", "latex_raw_bytes": nbytes}


def _raw_pdf_one(task: tuple[str, str, str, str, str, str]) -> dict:
    doc_id, pdf_path, zst_path, out_txt, server, meta_abstract = task
    global _LM
    try:
        tei = pp.grobid_fulltext(Path(pdf_path), server)
    except Exception as exc:
        return {"doc_id": doc_id, "pdf_raw_status": f"grobid_{type(exc).__name__}"}
    if not tei:
        return {"doc_id": doc_id, "pdf_raw_status": "grobid_empty"}
    doc = pp.tei_to_pes2o_text(tei, _LM, fallback_abstract=meta_abstract)
    text = (doc or {}).get("text", "")
    if not text:
        return {"doc_id": doc_id, "pdf_raw_status": "tei_empty"}
    st, nbytes = _verify_and_write_raw(text, zst_path, out_txt)
    return {"doc_id": doc_id, "pdf_raw_status": st,
            "pdf_raw_path": out_txt if st == "ok" else "", "pdf_raw_bytes": nbytes}


def cmd_rawtext(args: argparse.Namespace) -> int:
    rows = list(csv.DictReader(gzip.open(args.pool, "rt", newline="")))
    log(f"pool: {len(rows):,} papers; writing raw text for representations with status ok")
    raw_dir = args.out_dir / "raw"
    ltasks, ptasks = [], []
    for r in rows:
        if r.get("latex_status") == "ok" and r.get("latex_text_path"):
            dest = raw_dir / "latex" / r["yymm"] / f"{r['doc_id']}.txt"
            if dest.exists() and not args.force:
                r["latex_raw_status"], r["latex_raw_path"] = "ok", str(dest)
            else:
                ltasks.append((r["doc_id"], r["src_path"], r["latex_text_path"], str(dest)))
        if r.get("pdf_status") == "ok" and r.get("pdf_text_path"):
            dest = raw_dir / "pdf" / r["yymm"] / f"{r['doc_id']}.txt"
            if dest.exists() and not args.force:
                r["pdf_raw_status"], r["pdf_raw_path"] = "ok", str(dest)
            else:
                ptasks.append((r["doc_id"], r["pdf_path"], r["pdf_text_path"], str(dest),
                               args.server, r.get("abstract", "")))
    log(f"LaTeX raw: {len(ltasks):,} to write   PDF raw: {len(ptasks):,} to write")
    results: dict[str, dict] = {}
    ctx = mp.get_context("fork")
    if ltasks:
        with ctx.Pool(args.workers) as pool:
            for i, res in enumerate(pool.imap_unordered(_raw_latex_one, ltasks, chunksize=8), 1):
                results.setdefault(res["doc_id"], {}).update(res)
                if i % 1000 == 0:
                    log(f"  latex raw {i:,}/{len(ltasks):,}")
    if ptasks:
        with ctx.Pool(args.grobid_workers, initializer=_pdf_init,
                      initargs=(str(args.unigram_lm) if args.unigram_lm else None,)) as pool:
            for i, res in enumerate(pool.imap_unordered(_raw_pdf_one, ptasks, chunksize=1), 1):
                results.setdefault(res["doc_id"], {}).update(res)
                if i % 500 == 0:
                    log(f"  pdf raw {i:,}/{len(ptasks):,}")
    for r in rows:
        if r["doc_id"] in results:
            r.update({k: v for k, v in results[r["doc_id"]].items() if k != "doc_id"})
    cols = list(rows[0].keys())
    for extra in ("latex_raw_status", "latex_raw_path", "latex_raw_bytes",
                  "pdf_raw_status", "pdf_raw_path", "pdf_raw_bytes"):
        if extra not in cols:
            cols.append(extra)
    with gzip.open(args.pool, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    lat = Counter(r.get("latex_raw_status") or "n/a" for r in rows)
    pdf = Counter(r.get("pdf_raw_status") or "n/a" for r in rows)
    log(f"LaTeX raw status: {dict(lat.most_common(5))}")
    log(f"PDF raw status:   {dict(pdf.most_common(5))}")
    mism = sum(1 for r in rows if "raw_mismatch" in (r.get("latex_raw_status", ""), r.get("pdf_raw_status", "")))
    log(f"raw/normalized MISMATCHES (excluded downstream, .zst untouched): {mism:,}")
    log(f"updated {args.pool}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("sample", help="pick IDs per month and fetch metadata")
    s.add_argument("--arxiv-root", type=Path, required=True)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--months", nargs="+", required=True)
    s.add_argument("--per-month", type=int, default=1000)
    s.add_argument("--seed", type=int, default=20260915)
    s.add_argument("--api-delay", type=float, default=3.0)
    s.set_defaults(func=cmd_sample)

    q = sub.add_parser("parse", help="LaTeX + GROBID PDF, both normalized")
    q.add_argument("--pool", type=Path, required=True)
    q.add_argument("--out-dir", type=Path, required=True)
    q.add_argument("--server", default="http://localhost:8070")
    q.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    q.add_argument("--grobid-workers", type=int, default=8)
    q.add_argument("--unigram-lm", type=Path, default=None)
    q.add_argument("--force", action="store_true")
    q.set_defaults(func=cmd_parse)

    r = sub.add_parser("rawtext", help="write un-normalized text for injection/scoring; "
                                        "verifies it normalizes to the stored .zst")
    r.add_argument("--pool", type=Path, required=True, help="pool_parsed.csv.gz (updated in place)")
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--server", default="http://localhost:8070")
    r.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    r.add_argument("--grobid-workers", type=int, default=8)
    r.add_argument("--unigram-lm", type=Path, default=None)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_rawtext)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

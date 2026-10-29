#!/usr/bin/env python3
"""Reconstruct peS2o-style text from arXiv PDFs.

OLMo 2 ingested arXiv papers through two independent pipelines, and a
membership test built for one is systematically weak against the other:

* ``olmo-mix-1124/arxiv`` is RedPajama v1 -> Proof-Pile-2, cleaned from the
  **LaTeX source**. ``redpajama_arxiv_cleaner_standalone.py`` reproduces that
  byte-identically.
* ``pes2o`` is S2ORC, extracted from the **published PDF** with GROBID. Per the
  peS2o dataset card: "we use Grobid to extract section headers and paragraphs;
  figures, tables, and references, and any other non-textual content is
  removed." Equations are non-textual, so they vanish -- which is why a
  maths-heavy paper can score 5% verbatim containment against a pes2o record
  that is unmistakably the same paper.

This tool produces the second representation, so both sides of a comparison
come from the same kind of pipeline.

Honest about fidelity: this is NOT byte-identical to peS2o and cannot be.
peS2o used an unpinned GROBID version and then applied its own filters (English
only, >=500 words, >=5 paragraphs, sections dropped below an average log word
probability of -20, most-frequent-word constraints). ``verify`` measures the
actual agreement on papers whose pes2o record we already hold, so the gap is
reported rather than assumed.

Requires a running GROBID service:

    docker run -d --rm --name grobid -p 8070:8070 --init lfoppiano/grobid:0.8.1

Commands:

  parse    convert PDFs to peS2o-style text
  verify   compare parsed output against known pes2o records and report agreement
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import re
import sys
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from xml.etree import ElementTree as ET

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

TEI_NS = {"tei": "http://www.tei-c.org/ns/1.0"}
DEFAULT_GROBID = "http://localhost:8070"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# --------------------------------------------------------------------------
# GROBID
# --------------------------------------------------------------------------


def grobid_fulltext(pdf: Path, server: str, timeout: int = 300) -> str | None:
    """Return TEI XML for one PDF, or None on failure."""
    with pdf.open("rb") as handle:
        files = {"input": (pdf.name, handle, "application/pdf")}
        data = {
            # Match what S2ORC asks for: structured body, coordinates off.
            "consolidateHeader": "0",
            "consolidateCitations": "0",
            "segmentSentences": "0",
        }
        try:
            resp = requests.post(
                f"{server}/api/processFulltextDocument",
                files=files, data=data, timeout=timeout,
            )
        except Exception:
            return None
    if resp.status_code != 200 or not resp.text.strip():
        return None
    return resp.text


# --------------------------------------------------------------------------
# TEI -> peS2o-style text
# --------------------------------------------------------------------------


# Measured against 300 real pes2o records, not assumed. Median 8-gram recall
# of pes2o by configuration:
#
#   drop formula+figure+table+note+ref        0.829
#   + figure captions                         0.845
#   + table captions                          0.851
#   + KEEP citation refs                      0.944   <- decisive
#
# Citation markers were the dominant term: GROBID tags "[12]" as
# <ref type="bibr">, pes2o keeps it, and dropping it severs every n-gram
# spanning a citation. 70% of the n-grams we were missing contained digits,
# which is what pointed here. Formula blocks and notes contribute nothing and
# adding formulas back made recall WORSE, confirming pes2o excludes them.
INLINE_DROP = {"figure", "table", "graphic"}


def _text_of(node, drop=INLINE_DROP) -> str:
    """Flatten an element, dropping only genuinely non-textual children.

    Equations are excluded because GROBID emits <formula> as a block sibling
    of <p> rather than inline, and pes2o does not contain them -- that is the
    asymmetry that makes maths-heavy papers score low against pes2o.
    """
    parts: list[str] = []
    if node.text:
        parts.append(node.text)
    for child in node:
        tag = child.tag.split("}")[-1]
        if tag in drop:
            if child.tail:
                parts.append(child.tail)
            continue
        parts.append(_text_of(child, drop))
        if child.tail:
            parts.append(child.tail)
    return "".join(parts)


# ---------------------------------------------------------------------------
# Block-level quality filters
# ---------------------------------------------------------------------------
#
# peS2o removes "sections with average log word probability below -20". That
# threshold is defined against peS2o's own language model, which we do not
# have; calibrating an equivalent against a unigram model built FROM pes2o
# text puts the useful operating point near -9. Measured on 400 papers:
#
#   baseline                          Jaccard 0.8172   F1 0.9009
#   + log-prob filter (-9)                    0.8196      0.9028
#   + short-token filter                      0.8190      0.9022
#   + both                                    0.8209      0.9036   <- adopted
#   + log-prob at -8 (too aggressive)         0.7930      0.8880
#
# Gains are small because GROBID 0.8.1 produces little of the OCR garbage these
# filters exist to remove. They are kept because they mirror peS2o's described
# pipeline, which matters more for comparability than the 0.4 points.
#
# Filters that were tested and REJECTED for making agreement worse:
#   dropping bibliography-looking blocks   0.8172 -> 0.7663
#   dropping blocks under 3 words          0.8172 -> 0.8070
#   dropping "References"/"Acknowledgements" divs   no effect
#   de-duplicating repeated blocks                 no effect
MIN_MEAN_TOKEN_CHARS = 2.6
DEFAULT_MIN_LOGPROB = -9.0


def _mean_token_chars(block: str) -> float:
    w = block.split()
    return sum(len(x) for x in w) / len(w) if w else 0.0


def _avg_logprob(block: str, lm: dict, floor: float) -> float:
    w = re.sub(r"[^a-z0-9 ]+", " ", block.lower()).split()
    if not w:
        return 0.0
    return sum(lm.get(x, floor) for x in w) / len(w)


def filter_blocks(blocks: list, lm: dict | None, min_logprob: float) -> list:
    """Drop blocks that are symbol soup or improbable as natural language."""
    kept = []
    for b in blocks:
        if _mean_token_chars(b) < MIN_MEAN_TOKEN_CHARS:
            continue
        if lm and len(b.split()) >= 12:
            if _avg_logprob(b, lm["lm"], lm["floor"]) < min_logprob:
                continue
        kept.append(b)
    return kept



# GROBID sometimes promotes the first figure caption to <abstract> and drops
# the real abstract entirely (measured on 2509.11410: the TEI's abstract was
# "Figure 1: 3De lens inspecting wind turbine aerodynamics..." and the true
# abstract appeared nowhere in the document). peS2o records carry the real
# abstract, so a candidate parsed this way shares almost no 8-grams with its
# own corpus record and looks clean. When the caller can supply the paper's
# metadata abstract, it is used whenever GROBID's is missing, caption-like, or
# does not overlap it.
CAPTION_LIKE_RE = re.compile(r"^\s*(fig(ure)?|table|tab)\s*\.?\s*\d", re.I)
ABSTRACT_MIN_OVERLAP = 0.3


def _word_ngram_recall(reference: str, candidate: str, n: int = 8) -> float:
    """Fraction of `reference`'s word n-grams present in `candidate`."""
    ref = norm_hard(reference).split()
    cand = norm_hard(candidate).split()
    if len(ref) < n:
        return 1.0 if ref and " ".join(ref) in " ".join(cand) else 0.0
    grams = {tuple(ref[i:i + n]) for i in range(len(ref) - n + 1)}
    have = {tuple(cand[i:i + n]) for i in range(len(cand) - n + 1)}
    return len(grams & have) / len(grams)


def choose_abstract(grobid_abstract: str, fallback: str | None) -> tuple[str, str]:
    """(abstract, source). source is 'grobid' or 'metadata'."""
    fb = " ".join((fallback or "").split())
    if not fb:
        return grobid_abstract, "grobid"
    if not grobid_abstract:
        return fb, "metadata"
    if CAPTION_LIKE_RE.match(grobid_abstract):
        return fb, "metadata"
    if _word_ngram_recall(fb, grobid_abstract) < ABSTRACT_MIN_OVERLAP:
        return fb, "metadata"
    return grobid_abstract, "grobid"


def tei_to_pes2o_text(tei: str, lm: dict | None = None,
                      min_logprob: float = DEFAULT_MIN_LOGPROB,
                      fallback_abstract: str | None = None) -> dict:
    """Extract title, abstract and section-structured body from TEI.

    `fallback_abstract` (e.g. the arXiv metadata abstract) replaces GROBID's
    abstract when that is missing, a figure/table caption, or disjoint from it.
    """
    try:
        root = ET.fromstring(tei)
    except ET.ParseError:
        return {}

    title_el = root.find(".//tei:titleStmt/tei:title", TEI_NS)
    title = " ".join((title_el.text or "").split()) if title_el is not None else ""

    abstract_parts = []
    for p in root.findall(".//tei:profileDesc/tei:abstract//tei:p", TEI_NS):
        t = " ".join(_text_of(p).split())
        if t:
            abstract_parts.append(t)
    abstract, abstract_source = choose_abstract("\n".join(abstract_parts), fallback_abstract)

    def caption(fig) -> str:
        """Label + description, e.g. "Figure 1 Box plot of surgery duration".

        pes2o keeps the label, and emits the caption where the figure sits in
        the document rather than collected at the end.
        """
        head = fig.find("tei:head", TEI_NS)
        desc = fig.find("tei:figDesc", TEI_NS)
        bits = []
        for el in (head, desc):
            if el is not None:
                t = " ".join(_text_of(el).split())
                if t:
                    bits.append(t)
        return " ".join(bits)

    body_blocks: list[str] = []
    n_paragraphs = 0
    seen_figs = set()
    body_el = root.find(".//tei:text/tei:body", TEI_NS)
    for div in root.findall(".//tei:text/tei:body/tei:div", TEI_NS):
        head = div.find("tei:head", TEI_NS)
        if head is not None:
            h = " ".join(_text_of(head).split())
            if h:
                body_blocks.append(h)
        for child in div:
            tag = child.tag.split("}")[-1]
            if tag == "p":
                t = " ".join(_text_of(child).split())
                if t:
                    body_blocks.append(t)
                    n_paragraphs += 1
            elif tag == "figure":
                seen_figs.add(id(child))
                c = caption(child)
                if c:
                    body_blocks.append(c)

    # Figures that sit outside any div (GROBID often floats them to the end of
    # <body>) still belong in the text.
    if body_el is not None:
        for fig in body_el.findall("tei:figure", TEI_NS):
            if id(fig) in seen_figs:
                continue
            c = caption(fig)
            if c:
                body_blocks.append(c)

    body_blocks = filter_blocks(body_blocks, lm, min_logprob)
    n_paragraphs = min(n_paragraphs, len(body_blocks))

    body = "\n".join(body_blocks)
    # peS2o joins title, abstract and body with blank lines.
    text = "\n\n".join(x for x in (title, abstract, body) if x)
    return {
        "title": title,
        "abstract": abstract,
        "abstract_source": abstract_source,
        "body": body,
        "text": text,
        "n_paragraphs": n_paragraphs,
        "n_words": len(text.split()),
    }


# peS2o's published S2ORC filters, applied so the output is comparable.
def pes2o_filters(doc: dict, min_words: int = 500, min_paragraphs: int = 5) -> list:
    reasons = []
    if not doc.get("title"):
        reasons.append("no_title")
    if not doc.get("abstract"):
        reasons.append("no_abstract")
    if doc.get("n_words", 0) < min_words:
        reasons.append(f"under_{min_words}_words")
    if doc.get("n_paragraphs", 0) < min_paragraphs:
        reasons.append(f"under_{min_paragraphs}_paragraphs")
    return reasons


# --------------------------------------------------------------------------
# parse
# --------------------------------------------------------------------------


_LM: dict | None = None


def _one(task: tuple) -> dict:
    doc_id, pdf_path, out_dir, server, keep_tei, min_logprob = task
    out = Path(out_dir) / f"{doc_id.replace('/', '_')}.json.gz"
    if out.exists():
        return {"doc_id": doc_id, "status": "cached"}
    started = time.time()
    tei = grobid_fulltext(Path(pdf_path), server)
    if tei is None:
        return {"doc_id": doc_id, "status": "grobid_failed"}
    doc = tei_to_pes2o_text(tei, _LM, min_logprob)
    if not doc or not doc.get("text"):
        return {"doc_id": doc_id, "status": "empty_extraction"}
    doc["doc_id"] = doc_id
    doc["pdf"] = os.path.basename(pdf_path)
    doc["filters_failed"] = pes2o_filters(doc)
    doc["sha256"] = hashlib.sha256(doc["text"].encode()).hexdigest()
    doc["seconds"] = round(time.time() - started, 2)
    payload = json.dumps(doc, ensure_ascii=False).encode()
    atomic_write(out, gzip.compress(payload))
    if keep_tei:
        atomic_write(Path(out_dir) / "tei" / f"{doc_id}.tei.xml", tei.encode())
    return {"doc_id": doc_id, "status": "ok", "words": doc["n_words"],
            "seconds": doc["seconds"], "filters_failed": doc["filters_failed"]}


def cmd_parse(args: argparse.Namespace) -> int:
    if requests is None:
        raise SystemExit("pip install requests")
    alive = requests.get(f"{args.server}/api/isalive", timeout=10).text.strip()
    if "true" not in alive.lower():
        raise SystemExit(f"GROBID not alive at {args.server}")
    log(f"GROBID alive at {args.server}")
    global _LM
    if args.unigram_lm:
        _LM = json.loads(Path(args.unigram_lm).read_text())
        log(f"unigram LM: {len(_LM['lm']):,} words, floor {_LM['floor']:.2f}, "
            f"threshold {args.min_logprob}")
    else:
        log("no unigram LM given; log-probability block filter disabled")

    tasks = []
    with open(args.input_list) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc_id, pdf = line.split("\t")[:2]
            tasks.append((doc_id, pdf, str(args.out_dir), args.server,
                          args.keep_tei, args.min_logprob))
    log(f"{len(tasks):,} PDFs to parse with {args.workers} concurrent requests")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.time()
    counts: dict[str, int] = {}
    done = 0
    with open(args.out_dir / "parse_log.jsonl", "a") as journal:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for res in pool.map(_one, tasks):
                done += 1
                counts[res["status"]] = counts.get(res["status"], 0) + 1
                journal.write(json.dumps(res) + "\n")
                if done % 25 == 0 or done == len(tasks):
                    rate = done / max(time.time() - started, 1e-9)
                    log(f"  {done:,}/{len(tasks):,}  {rate:.2f} PDF/s  "
                        f"eta {(len(tasks) - done) / max(rate, 1e-9) / 60:.1f} min  {counts}")
    summary = {
        "pdfs": len(tasks),
        "status_counts": counts,
        "elapsed_seconds": round(time.time() - started, 1),
        "pdf_per_second": round(len(tasks) / max(time.time() - started, 1e-9), 3),
        "grobid": args.server,
    }
    (args.out_dir / "parse_summary.json").write_text(json.dumps(summary, indent=2))
    log(json.dumps(summary))
    return 0


# --------------------------------------------------------------------------
# verify
# --------------------------------------------------------------------------


def norm_ws(t: str) -> str:
    return " ".join(t.split())


def norm_hard(t: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t.lower()).split())


def ngrams(t: str, n: int = 8) -> set:
    w = t.split()
    return {" ".join(w[i:i + n]) for i in range(max(len(w) - n + 1, 0))}


def edit_similarity(a: str, b: str, cap: int = 6000) -> float:
    """Word-level alignment ratio (an edit-distance style score).

    Character Levenshtein is O(n*m) and infeasible on full papers, so this
    aligns word sequences with difflib and reports 2*matches/(len_a+len_b),
    capped for tractability.
    """
    import difflib
    aw, bw = a.split()[:cap], b.split()[:cap]
    if not aw or not bw:
        return 0.0
    return difflib.SequenceMatcher(None, aw, bw, autojunk=False).ratio()


def cmd_verify(args: argparse.Namespace) -> int:
    ours = {}
    for path in Path(args.parsed_dir).glob("*.json.gz"):
        d = json.loads(gzip.decompress(path.read_bytes()))
        ours[d["doc_id"]] = d
    truth = json.loads(Path(args.pes2o_records).read_text())
    pairs = json.loads(Path(args.pairs).read_text())

    rows = []
    for item in pairs:
        did, ref = item["doc_id"], str(item["pes2o_ref"])
        o, t = ours.get(did), truth.get(ref)
        if not o or not t:
            continue
        ot, tt = o["text"], t.get("text", "")
        og, tg = ngrams(norm_hard(ot)), ngrams(norm_hard(tt))
        inter = len(og & tg)
        # Different arXiv versions can carry different titles; that is a
        # version mismatch, not a parsing failure, and must not be counted as
        # one when reporting fidelity.
        ot_title = norm_hard(ot.split("\n")[0])
        tt_title = norm_hard(tt.split("\n")[0])
        tset, oset = set(tt_title.split()), set(ot_title.split())
        title_overlap = len(tset & oset) / max(len(tset), 1)
        # Content agreement is itself proof of identity and outranks the title:
        # papers are routinely retitled between arXiv versions, and validation
        # found cases at 0.96 recall whose titles shared under half their words.
        rows.append({
            "doc_id": did,
            "title_match": round(title_overlap, 3),
            "same_version": title_overlap >= 0.6 or inter / max(len(tg), 1) >= 0.8,
            "edit_similarity": round(edit_similarity(norm_hard(ot), norm_hard(tt)), 4),
            "exact_bytes": ot == tt,
            "exact_whitespace_normalized": norm_ws(ot) == norm_ws(tt),
            "exact_hard_normalized": norm_hard(ot) == norm_hard(tt),
            "our_words": len(ot.split()),
            "pes2o_words": len(tt.split()),
            "word_ratio": round(len(ot.split()) / max(len(tt.split()), 1), 3),
            "ngram_recall_of_pes2o": round(inter / max(len(tg), 1), 4),
            "ngram_precision": round(inter / max(len(og), 1), 4),
            "jaccard": round(inter / max(len(og | tg), 1), 4),
        })

    if not rows:
        raise SystemExit("no comparable pairs found")
    import statistics as st

    def med(k):
        return round(st.median([r[k] for r in rows]), 4)

    same = [r for r in rows if r["same_version"]]
    summary = {
        "pairs_compared": len(rows),
        "same_version_pairs": len(same),
        "version_mismatch_pairs": len(rows) - len(same),
        "median_recall_same_version_only": (
            round(st.median([r["ngram_recall_of_pes2o"] for r in same]), 4) if same else None),
        "median_edit_similarity_same_version_only": (
            round(st.median([r["edit_similarity"] for r in same]), 4) if same else None),
        "exact_byte_matches": sum(r["exact_bytes"] for r in rows),
        "exact_after_whitespace_norm": sum(r["exact_whitespace_normalized"] for r in rows),
        "exact_after_hard_norm": sum(r["exact_hard_normalized"] for r in rows),
        "median_edit_similarity": med("edit_similarity"),
        "median_ngram_recall_of_pes2o": med("ngram_recall_of_pes2o"),
        "median_jaccard": med("jaccard"),
        "median_word_ratio": med("word_ratio"),
        "recall_at_least_0.5": sum(r["ngram_recall_of_pes2o"] >= 0.5 for r in rows),
        "recall_at_least_0.8": sum(r["ngram_recall_of_pes2o"] >= 0.8 for r in rows),
        "recall_at_least_0.95": sum(r["ngram_recall_of_pes2o"] >= 0.95 for r in rows),
    }
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verify_summary.json").write_text(json.dumps(summary, indent=2))
    with gzip.open(out / "verify_rows.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(json.dumps(summary, indent=2))
    return 0


def cmd_build_lm(args: argparse.Namespace) -> int:
    """Unigram log-probability table, built from real pes2o text.

    peS2o's -20 threshold is defined against its own model; building ours from
    pes2o itself keeps the scale interpretable and the filter reproducible.
    """
    import collections
    import math

    records = json.loads(Path(args.pes2o_records).read_text())
    counts: collections.Counter = collections.Counter()
    for rec in records.values():
        text = unicodedata.normalize("NFKD", rec.get("text", ""))
        counts.update(re.sub(r"[^a-z0-9 ]+", " ", text.lower()).split())
    total = sum(counts.values())
    lm = {w: math.log(n / total) for w, n in counts.items() if n >= 3}
    payload = {"lm": lm, "floor": math.log(0.5 / total), "total": total,
               "vocab": len(lm), "records": len(records)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))
    log(f"unigram LM from {len(records):,} records: {total:,} tokens, "
        f"vocab {len(lm):,}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("parse", help="convert PDFs to peS2o-style text")
    a.add_argument("--input-list", required=True,
                   help="TSV: doc_id <TAB> /path/to.pdf per line")
    a.add_argument("--out-dir", type=Path, required=True)
    a.add_argument("--server", default=DEFAULT_GROBID)
    a.add_argument("--workers", type=int, default=8)
    a.add_argument("--keep-tei", action="store_true")
    a.add_argument("--unigram-lm", default=None,
                   help="JSON from build-lm; enables peS2o's section-quality filter")
    a.add_argument("--min-logprob", type=float, default=DEFAULT_MIN_LOGPROB)
    a.set_defaults(func=cmd_parse)

    b = sub.add_parser("build-lm", help="unigram LM for the quality filter")
    b.add_argument("--pes2o-records", required=True, help="JSON: ref -> record")
    b.add_argument("--out", type=Path, required=True)
    b.set_defaults(func=cmd_build_lm)

    v = sub.add_parser("verify", help="measure agreement with real pes2o records")
    v.add_argument("--parsed-dir", required=True)
    v.add_argument("--pes2o-records", required=True, help="JSON: ref -> record")
    v.add_argument("--pairs", required=True, help="JSON list of {doc_id, pes2o_ref}")
    v.add_argument("--out-dir", type=Path, required=True)
    v.set_defaults(func=cmd_verify)
    return p


def main() -> int:
    args = build_parser().parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the post-cutoff StackExchange candidate pool (cohort S-se).

Source: the archive.org StackExchange dump, one `<site>.7z` per site, each
holding Posts.xml. Dolma's `stackexchange` component (dolmino-mix-1124) is
ONE record per (question, answer) pair -- id `{forum}-{qid}-{aid}` -- with the
text rendered by stackexchange_dolma_text.record_text, which reproduces the
corpus text exactly after normalization (519/519 pairs) and byte for byte on
77.6% (the rest differ only in blank-line count at the title/Q/A joins, a
per-record quirk of Dolma's snapshot).

Non-membership by construction requires BOTH the question and the answer to
be created after the cutoff: a pre-cutoff question already appears in the
corpus paired with its older answers, so its text is training data even when
the answer is new.

Subcommands
    build       parse Posts.xml per site, keep post-cutoff pairs, sample per
                month, render, normalize, write the manifest
    calibrate   the fidelity measurement: pairs present in both our dump and
                dolmino with identical LastEditDate, byte / normalized identity
"""

from __future__ import annotations

import argparse
import csv
import gzip
import html
import json
import os
import random
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

import zstandard as zstd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import olmo2_midtrain_membership as mm  # noqa: E402
import stackexchange_dolma_text as sdt  # noqa: E402

CUTOFF = "2025-04-17"
MIN_WORDS = 50


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def forum_name(site_archive: str) -> str:
    """'math.stackexchange.com.7z' -> 'math_stackexchange_com' (Dolma's forum key)."""
    base = site_archive[:-3] if site_archive.endswith(".7z") else site_archive
    base = base.replace("-Posts", "")
    return base.replace(".", "_")


def extract_posts_xml(archive: Path, workdir: Path) -> Path:
    import py7zr
    out = workdir / (archive.stem + "_Posts.xml")
    if out.exists() and out.stat().st_size > 0:
        return out
    with py7zr.SevenZipFile(archive) as z:
        z.extract(path=workdir, targets=["Posts.xml"])
    (workdir / "Posts.xml").rename(out)
    return out


def iter_posts(posts_xml: Path):
    for _ev, el in ET.iterparse(posts_xml, events=("end",)):
        if el.tag == "row":
            yield el.attrib
            el.clear()


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def cmd_build(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    texts_dir = args.out_dir / "texts"
    texts_dir.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="se_posts_", dir=args.scratch))
    rng = random.Random(args.seed)
    cctx = zstd.ZstdCompressor(level=6)
    rows: list[dict] = []
    cols = ["doc_id", "forum", "month", "q_id", "a_id", "q_created", "a_created",
            "a_score", "accepted", "n_words", "norm_bytes", "text_path",
            "raw_text_path", "parser_status"]

    for site in args.sites:
        archive = args.dump_dir / site
        if not archive.exists():
            log(f"SKIP {site}: not downloaded")
            continue
        forum = forum_name(site)
        log(f"{forum}: extracting Posts.xml")
        posts_xml = extract_posts_xml(archive, work)

        # Pass 1: post-cutoff questions.
        questions: dict[str, dict] = {}
        for p in iter_posts(posts_xml):
            if p.get("PostTypeId") == "1" and p.get("CreationDate", "") >= CUTOFF:
                questions[p["Id"]] = p
        # Pass 2: their post-cutoff answers.
        pairs: dict[str, list[dict]] = defaultdict(list)  # month -> candidate pairs
        for p in iter_posts(posts_xml):
            if p.get("PostTypeId") != "2":
                continue
            q = questions.get(p.get("ParentId", ""))
            if q is None or p.get("CreationDate", "") < CUTOFF:
                continue
            accepted = q.get("AcceptedAnswerId") == p["Id"]
            score = int(p.get("Score", "0") or 0)
            if not accepted and score < args.min_score:
                continue
            pairs[q["CreationDate"][:7]].append({"q": q, "a": p, "accepted": accepted, "score": score})
        log(f"{forum}: {len(questions):,} post-cutoff questions; candidate pairs by month: "
            f"{ {m: len(v) for m, v in sorted(pairs.items())} }")

        # Pass 3: sample per month, render, store.
        kept = short = 0
        for month in sorted(pairs):
            pop = sorted(pairs[month], key=lambda x: (int(x["q"]["Id"]), int(x["a"]["Id"])))
            for item in rng.sample(pop, min(args.per_month, len(pop))):
                q, a = item["q"], item["a"]
                text = sdt.record_text(html.unescape(q.get("Title", "")), q["Body"], a["Body"])
                nw = len(text.split())
                if nw < MIN_WORDS:
                    short += 1
                    continue
                norm = mm.normalize(text)
                doc_id = f"{forum}-{q['Id']}-{a['Id']}"
                raw_path = texts_dir / f"{doc_id}.txt"
                zst_path = texts_dir / f"{doc_id}.zst"
                raw_path.write_text(text, encoding="utf-8")
                zst_path.write_bytes(cctx.compress(norm))
                rows.append({
                    "doc_id": f"se:{doc_id}", "forum": forum, "month": month,
                    "q_id": q["Id"], "a_id": a["Id"], "q_created": q["CreationDate"],
                    "a_created": a["CreationDate"], "a_score": item["score"],
                    "accepted": int(item["accepted"]), "n_words": nw, "norm_bytes": len(norm),
                    "text_path": str(zst_path), "raw_text_path": str(raw_path),
                    "parser_status": "ok",
                })
                kept += 1
        log(f"{forum}: kept {kept:,}  (short {short})")
        posts_xml.unlink(missing_ok=True)

    out = args.out_dir / "se_pool.csv.gz"
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    per = Counter((r["forum"], r["month"]) for r in rows)
    log(f"\nTOTAL {len(rows):,} (question, answer) pairs")
    for f in sorted({k[0] for k in per}):
        log(f"  {f}: " + " ".join(f"{m[-2:]}:{per[(f, m)]}" for m in sorted(k[1] for k in per if k[0] == f)))
    log(f"wrote {out}")
    return 0


# --------------------------------------------------------------------------
# calibrate
# --------------------------------------------------------------------------


def cmd_calibrate(args: argparse.Namespace) -> int:
    import difflib
    forum = forum_name(args.site)
    work = Path(tempfile.mkdtemp(prefix="se_cal_", dir=args.scratch))
    posts_xml = extract_posts_xml(args.dump_dir / args.site, work)
    posts = {p["Id"]: p for p in iter_posts(posts_xml)}
    log(f"{forum}: {len(posts):,} posts in our dump")

    shards = mm.discover_shards(args.corpus, ["stackexchange"])
    recs = []
    needle = forum.encode()
    for s in shards:
        for raw in mm.iter_lines(s.path):
            if needle not in raw:
                continue
            r = json.loads(raw)
            if r["metadata"].get("forum") == forum:
                recs.append(r)
        if len(recs) >= args.max_records:
            break
    log(f"{forum}: {len(recs):,} dolmino records collected")

    ident = norm_ident = usable = skipped = 0
    sims = []
    residual = Counter()
    for r in recs:
        md = r["metadata"]
        q = posts.get(str(md["question_id"]))
        a = posts.get(str(md["answer_id"]))
        if not q or not a:
            skipped += 1
            continue
        if (q.get("LastEditDate") or "") != (md.get("question_last_edit_date") or "") or \
           (a.get("LastEditDate") or "") != (md.get("answer_last_edit_date") or ""):
            skipped += 1
            continue
        usable += 1
        ours = sdt.record_text(html.unescape(q.get("Title", "")), q["Body"], a["Body"])
        ref = r["text"]
        if ours == ref:
            ident += 1
        else:
            residual["newline_count_only" if ours.split() == ref.split() else "other"] += 1
        if mm.normalize(ours) == mm.normalize(ref):
            norm_ident += 1
        sims.append(difflib.SequenceMatcher(None, ours, ref, autojunk=False).ratio())
    log(f"\npairs present in both, unchanged since Dolma's snapshot: {usable:,} (skipped {skipped:,})")
    log(f"  byte-identical:              {ident:,}  ({100*ident/max(usable,1):.1f}%)")
    log(f"  identical after normalize:   {norm_ident:,}  ({100*norm_ident/max(usable,1):.1f}%)  <- what the scan compares")
    log(f"  residual classes: {dict(residual)}")
    log(f"  mean similarity {sum(sims)/max(len(sims),1):.5f}   min {min(sims) if sims else 0:.4f}")
    posts_xml.unlink(missing_ok=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--dump-dir", type=Path, required=True)
    b.add_argument("--sites", nargs="+", required=True, help="e.g. math.stackexchange.com.7z")
    b.add_argument("--out-dir", type=Path, required=True)
    b.add_argument("--per-month", type=int, default=300)
    b.add_argument("--min-score", type=int, default=1,
                   help="answers below this score are kept only if accepted")
    b.add_argument("--seed", type=int, default=20260915)
    b.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    b.set_defaults(func=cmd_build)
    c = sub.add_parser("calibrate")
    c.add_argument("--dump-dir", type=Path, required=True)
    c.add_argument("--site", required=True)
    c.add_argument("--corpus", type=Path, required=True, help="dolmino-mix-1124 root")
    c.add_argument("--max-records", type=int, default=2000)
    c.add_argument("--scratch", type=Path, default=Path(tempfile.gettempdir()))
    c.set_defaults(func=cmd_calibrate)
    a = ap.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())

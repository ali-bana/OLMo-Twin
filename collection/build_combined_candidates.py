#!/usr/bin/env python3
"""Merge the post-cutoff pools into one candidate manifest for the audit scan.

Everything is scanned in a single corpus pass, because reading the 8.3 TiB of
training data dominates the cost -- adding rows is nearly free, re-reading the
corpus is not.

arXiv papers appear TWICE, once per representation, because the corpora ingest
arXiv through two different pipelines and a paper can be invisible in one while
plainly present in the other:

    <id>#latex   RedPajama-style cleaned LaTeX source -> matches olmo-mix `arxiv`
    <id>#pdf     GROBID/peS2o-style PDF text          -> matches `pes2o`

Results are unioned back per paper afterwards: a hit in EITHER representation
means the paper is a member.

Wikipedia rows carry their group:
    wikiA:<pageid>   created after the release date  (non-member by construction)
    wikiB:<pageid>   pre-existing, edited after it   (expected to overlap)
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import sys
from collections import Counter
from pathlib import Path

CSV_COLUMNS = [
    "doc_id", "source", "rep", "group", "month", "category_group",
    "title", "abstract", "text_path", "parser_status",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arxiv-pool", type=Path, required=True)
    ap.add_argument("--wiki-pool", type=Path, required=True)
    ap.add_argument("--se-pool", type=Path, default=None,
                    help="se_pool.csv.gz from build_stackexchange_pool.py (optional)")
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    csv.field_size_limit(sys.maxsize)
    rows: list[dict] = []

    n_arxiv_papers = 0
    both = 0
    for r in csv.DictReader(gzip.open(args.arxiv_pool, "rt", newline="")):
        n_arxiv_papers += 1
        ok_latex = r.get("latex_status") == "ok" and r.get("latex_text_path")
        ok_pdf = r.get("pdf_status") == "ok" and r.get("pdf_text_path")
        if ok_latex and ok_pdf:
            both += 1
        for rep, path in (("latex", r.get("latex_text_path")),
                          ("pdf", r.get("pdf_text_path"))):
            status = r.get(f"{rep}_status")
            if status != "ok" or not path:
                continue
            rows.append({
                "doc_id": f"{r['doc_id']}#{rep}", "source": "arxiv", "rep": rep,
                "group": "postcutoff", "month": r["month"],
                "category_group": r.get("category_group", ""),
                "title": r.get("title", ""), "abstract": r.get("abstract", ""),
                "text_path": path, "parser_status": "ok",
            })

    n_wiki = 0
    for r in csv.DictReader(gzip.open(args.wiki_pool, "rt", newline="")):
        if r.get("parser_status") != "ok":
            continue
        n_wiki += 1
        rows.append({
            "doc_id": r["doc_id"], "source": "wikipedia", "rep": "text",
            "group": f"wiki{r['group']}", "month": r["month"],
            "category_group": "", "title": r.get("title", ""), "abstract": "",
            "text_path": r["text_path"], "parser_status": "ok",
        })

    n_se = 0
    if args.se_pool:
        for r in csv.DictReader(gzip.open(args.se_pool, "rt", newline="")):
            if r.get("parser_status") != "ok":
                continue
            n_se += 1
            rows.append({
                "doc_id": r["doc_id"], "source": "stackexchange", "rep": "text",
                "group": "se", "month": r["month"],
                "category_group": r.get("forum", ""), "title": "", "abstract": "",
                "text_path": r["text_path"], "parser_status": "ok",
            })

    out = args.out_dir / "combined.csv.gz"
    with gzip.open(out, "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    # The scan reports hits by integer index into the parser_status=="ok" rows,
    # in order; persist that mapping so results can be attributed later.
    (args.out_dir / "index_ids.json").write_text(
        json.dumps([r["doc_id"] for r in rows])
    )

    by_src = Counter((r["source"], r["rep"], r["group"]) for r in rows)
    print(f"arXiv papers in pool:            {n_arxiv_papers:,}")
    print(f"  with BOTH representations ok:  {both:,}")
    print(f"wikipedia articles:              {n_wiki:,}")
    print(f"stackexchange (Q,A) pairs:       {n_se:,}")
    print(f"\ncombined rows (fingerprint sets): {len(rows):,}")
    for k in sorted(by_src):
        print(f"   {k[0]:<10} {k[1]:<6} {k[2]:<12} {by_src[k]:>7,}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

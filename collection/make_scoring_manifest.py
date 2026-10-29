#!/usr/bin/env python3
"""Join the Stage-G clean set with the raw-text paths of the candidate pools.

Writes audit_v2/stageG/scoring_manifest.csv.gz: every row of
candidates_clean.csv.gz (assemble_injection_candidates.py) plus
`raw_text_path` and `text_form`, the un-normalized text that is scored under the
pre-treatment checkpoint (scoring/score_pretreatment_logprobs.py) and later
injected. For arXiv the PDF form is preferred (`pes2o_style_pdf`, written by
`build_postcutoff_arxiv_pool.py rawtext`), else the LaTeX form
(`redpajama_latex`); Wikipedia rows take the Dolma-format text of the pool
(`dolma_wiki`) and StackExchange rows that of the SE pool (`dolma_se`; the
paper's SE pool came from fetch_stackexchange_api.py). Clean documents without
raw text are dropped and counted.

The code is the inline step of the original collection driver, unchanged
except that the five paths are arguments.
"""
import argparse
import csv
import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402


def main() -> int:
    C = paths.COLLECTION
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arxiv-pool", type=Path, default=C / "postcutoff" / "pool_parsed_merged.csv.gz",
                    help="merged arXiv pool after rawtext (default: collection/postcutoff/pool_parsed_merged.csv.gz)")
    ap.add_argument("--wiki-pool", type=Path, default=C / "wikipools_v2" / "wiki_pools.csv.gz",
                    help="Wikipedia pool (default: collection/wikipools_v2/wiki_pools.csv.gz)")
    ap.add_argument("--se-pool", type=Path, default=C / "sepool_api" / "se_pool.csv.gz",
                    help="StackExchange pool (default: collection/sepool_api/se_pool.csv.gz)")
    ap.add_argument("--clean", type=Path, default=C / "audit_v2" / "stageG" / "candidates_clean.csv.gz",
                    help="Stage-G clean set (default: collection/audit_v2/stageG/candidates_clean.csv.gz)")
    ap.add_argument("--out", type=Path, default=C / "audit_v2" / "stageG" / "scoring_manifest.csv.gz",
                    help="output (default: collection/audit_v2/stageG/scoring_manifest.csv.gz)")
    args = ap.parse_args()

    csv.field_size_limit(sys.maxsize)
    raw={}
    for r in csv.DictReader(gzip.open(args.arxiv_pool,"rt",newline="")):
        p=r.get("pdf_raw_path") or ""
        if r.get("pdf_raw_status")=="ok" and p: raw[r["doc_id"]]=(p,"pes2o_style_pdf")
        elif r.get("latex_raw_status")=="ok" and r.get("latex_raw_path"): raw[r["doc_id"]]=(r["latex_raw_path"],"redpajama_latex")
    for r in csv.DictReader(gzip.open(args.wiki_pool,"rt",newline="")):
        raw[r["doc_id"].split(":",1)[1]]=(r["raw_text_path"],"dolma_wiki")
    for r in csv.DictReader(gzip.open(args.se_pool,"rt",newline="")):
        raw[r["doc_id"]]=(r["raw_text_path"],"dolma_se")
    rows=list(csv.DictReader(gzip.open(args.clean,"rt",newline="")))
    out=[]; miss=0
    for r in rows:
        key=r["doc_id"].split(":",1)[1] if r["doc_id"].startswith("wiki") else r["doc_id"]
        hit=raw.get(key)
        if not hit: miss+=1; continue
        out.append({**r,"raw_text_path":hit[0],"text_form":hit[1]})
    with gzip.open(args.out,"wt",newline="") as fh:
        w=csv.DictWriter(fh,fieldnames=list(out[0].keys())); w.writeheader(); [w.writerow(r) for r in out]
    print(f"   scoring manifest: {len(out):,} clean documents with raw text ({miss} without raw text, excluded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

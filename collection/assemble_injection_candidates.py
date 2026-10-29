#!/usr/bin/env python3
"""Stage G: turn scan + vector-search evidence into the injection candidate set.

A document is EXCLUDED (not a usable non-member) if ANY of these fires:

  span      longest contiguous run shared with one corpus record >= --span-words
            (the "verbatim span" rule; catches reused blocks)
  count     distinct shared 8-gram shingles with one record >= --shingles
            (catches diffuse / paraphrased reuse a span rule misses)
  semantic  cosine to a pes2o opening >= --cosine
            (catches rewording that defeats n-grams; 4-5 papers in this pool)
  no_fp     the document produced no fingerprints at all -- it is untestable
  unverified (Wikipedia) the page's first revision could not be established
  ancestor  (Wikipedia, with --wiki-ancestors) the page's TRAINED version is a
            corpus record, so the page is a member however heavily it was
            edited since; the n-gram tests miss ~4% of these (rewrites)

arXiv papers appear in the manifest twice (LaTeX and PDF representations);
a paper is excluded if EITHER representation trips a test. Wikipedia group
labels are taken from the verified first-revision file, never from the
manifest. Provenance digests recorded by the scan are checked against the
fingerprint metadata and the manifest before anything is read.

Everything the decision rests on is written per document (best shingles,
best run, best cosine, which component, which corpus record), so every
exclusion -- and every inclusion -- can be audited.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def log(msg: str) -> None:
    print(msg, flush=True)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def paper_key(doc_id: str) -> str:
    return doc_id.split("#")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", type=Path, required=True,
                    help="dir with combined.csv.gz, index_ids.json, fp/, scan_*/")
    ap.add_argument("--embed-matches", type=Path, required=True,
                    help="embed_matches.jsonl.gz from the vector search")
    ap.add_argument("--wiki-labels", type=Path, required=True,
                    help="wiki_pools_verified.csv.gz (true first-revision groups)")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--span-words", type=int, default=40,
                    help="contiguous shared words that exclude (~50 OLMo tokens)")
    ap.add_argument("--shingles", type=int, default=50)
    ap.add_argument("--cosine", type=float, default=0.90)
    ap.add_argument("--wiki-ancestors", type=Path, default=None,
                    help="ancestors.jsonl.gz from build_edited_wikipedia_group.py: a Wikipedia page whose "
                         "trained version exists in the corpus is excluded (reason 'ancestor') whether "
                         "or not the n-gram tests catch the edited text")
    ap.add_argument("--allow-missing-run", action="store_true",
                    help="accept hit records without run_words (scans made before "
                         "position tracking existed); the span test is then skipped "
                         "and the output says so")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    csv.field_size_limit(sys.maxsize)
    A = args.audit_dir

    # ---- provenance -------------------------------------------------------
    meta = json.loads((A / "fp" / "fingerprint_meta.json").read_text())
    manifest_digest = sha256_file(A / "combined.csv.gz")
    if meta.get("candidates_sha256") and meta["candidates_sha256"] != manifest_digest:
        raise SystemExit("combined.csv.gz is not the manifest the fingerprints were built from")
    scan_dirs = sorted(d for d in glob.glob(str(A / "scan_*"))
                       if os.path.exists(os.path.join(d, "scan_summary.json")))
    if not scan_dirs:
        raise SystemExit("no completed scan targets under the audit dir")
    fp_digests = set()
    for d in scan_dirs:
        s = json.loads(Path(d, "scan_summary.json").read_text())
        if s.get("fingerprints_sha256"):
            fp_digests.add(s["fingerprints_sha256"])
        if s.get("suspect_shards"):
            raise SystemExit(f"{os.path.basename(d)} has suspect shards; refusing")
    if len(fp_digests) > 1 or (fp_digests and meta.get("fingerprints_sha256")
                               and fp_digests != {meta["fingerprints_sha256"]}):
        raise SystemExit("scan targets were not all produced from the same fingerprint table")
    log(f"provenance ok: {len(scan_dirs)} scan targets, manifest {manifest_digest[:12]}")

    # ---- manifest ---------------------------------------------------------
    ids = json.loads((A / "index_ids.json").read_text())
    rows = list(csv.DictReader(gzip.open(A / "combined.csv.gz", "rt", newline="")))
    assert len(ids) == len(rows) and all(r["doc_id"] == d for r, d in zip(rows, ids)), \
        "index_ids.json does not match combined.csv.gz row for row"

    # documents that produced no fingerprints (untestable)
    owners = np.load(A / "fp" / "owners.npy", mmap_mode="r")
    has_fp = set(np.unique(np.asarray(owners)).tolist())

    wiki_group: dict[str, str] = {}
    for r in csv.DictReader(gzip.open(args.wiki_labels, "rt", newline="")):
        wiki_group[r["doc_id"].split(":", 1)[1]] = r["group"]
    wiki_ancestor: dict[str, str] = {}   # page id -> corpus record id of the trained version
    if args.wiki_ancestors:
        for line in gzip.open(args.wiki_ancestors, "rt"):
            a = json.loads(line)
            wiki_ancestor[a["page_id"]] = a.get("ancestor_record_id", a["page_id"])
        log(f"wiki ancestors: {len(wiki_ancestor):,} pages have their trained version in the corpus")

    # ---- n-gram evidence: best per document index ---------------------------
    best = {}  # idx -> dict(shingles, run_words, component, ref)
    have_run = True
    for d in scan_dirs:
        comp = os.path.basename(d).replace("scan_", "")
        for f in glob.glob(os.path.join(d, "shards", "*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    try:
                        e = json.loads(line)
                    except Exception:
                        continue
                    i = e.get("doc")
                    if i is None:
                        continue
                    for m in e.get("matches", []):
                        if "run_words" not in m:
                            have_run = False
                        sh = int(m.get("shingles", 0))
                        rw = int(m.get("run_words", 0))
                        cur = best.setdefault(i, {"shingles": 0, "run_words": 0,
                                                  "component": "", "ref": "",
                                                  "run_component": "", "run_ref": ""})
                        if sh > cur["shingles"]:
                            cur.update(shingles=sh, component=comp, ref=str(m.get("ref", "")))
                        if rw > cur["run_words"]:
                            cur.update(run_words=rw, run_component=comp, run_ref=str(m.get("ref", "")))
    if not have_run and not args.allow_missing_run:
        raise SystemExit(
            "hit records lack run_words (scan predates position tracking). The span "
            "test cannot be applied. Re-scan, or pass --allow-missing-run to proceed "
            "with count + semantic tests only."
        )
    span_applied = have_run

    # ---- semantic evidence ------------------------------------------------
    cos = {}  # doc_id -> (cosine, ref)
    with gzip.open(args.embed_matches, "rt") as fh:
        for line in fh:
            e = json.loads(line)
            ms = e.get("matches", [])
            if ms:
                top = max(ms, key=lambda m: float(m.get("cosine", 0)))
                cos[e["doc_id"]] = (float(top["cosine"]), str(top.get("ref", "")))

    # ---- collapse to documents (arXiv: union of representations) -----------
    docs: dict[str, dict] = {}
    for i, r in enumerate(rows):
        key = paper_key(r["doc_id"])
        grp = r["group"]
        if grp.startswith("wiki"):
            pid = r["doc_id"].split(":", 1)[1]
            g = wiki_group.get(pid)
            grp = f"wiki{g}" if g else "wiki_unverified"
        d = docs.setdefault(key, {
            "doc_id": key, "source": r["source"], "group": grp, "month": r["month"],
            "category_group": r.get("category_group", ""), "title": r.get("title", ""),
            "reps": 0, "reps_with_fp": 0,
            "best_shingles": 0, "shingles_component": "", "shingles_ref": "",
            "best_run_words": 0, "run_component": "", "run_ref": "",
            "best_cosine": 0.0, "cosine_ref": "",
            "ancestor_ref": wiki_ancestor.get(r["doc_id"].split(":", 1)[1], "") if grp.startswith("wiki") else "",
        })
        d["reps"] += 1
        if i in has_fp:
            d["reps_with_fp"] += 1
        b = best.get(i)
        if b:
            if b["shingles"] > d["best_shingles"]:
                d.update(best_shingles=b["shingles"], shingles_component=b["component"],
                         shingles_ref=b["ref"])
            if b["run_words"] > d["best_run_words"]:
                d.update(best_run_words=b["run_words"], run_component=b["run_component"],
                         run_ref=b["run_ref"])
        c = cos.get(r["doc_id"])
        if c and c[0] > d["best_cosine"]:
            d.update(best_cosine=c[0], cosine_ref=c[1])

    # ---- decide -----------------------------------------------------------
    for d in docs.values():
        reasons = []
        if d["group"] == "wiki_unverified":
            reasons.append("unverified_date")
        if d["reps_with_fp"] == 0:
            reasons.append("no_fingerprints")
        if span_applied and d["best_run_words"] >= args.span_words:
            reasons.append("span")
        if d["best_shingles"] >= args.shingles:
            reasons.append("count")
        if d["best_cosine"] >= args.cosine:
            reasons.append("semantic")
        if d["ancestor_ref"]:
            reasons.append("ancestor")
        d["excluded"] = bool(reasons)
        d["reasons"] = "|".join(reasons)

    # ---- write --------------------------------------------------------------
    cols = ["doc_id", "source", "group", "month", "category_group", "title", "reps",
            "reps_with_fp", "best_shingles", "shingles_component", "shingles_ref",
            "best_run_words", "run_component", "run_ref", "best_cosine", "cosine_ref",
            "ancestor_ref", "excluded", "reasons"]
    all_path = args.out_dir / "candidates_all_evidence.csv.gz"
    clean_path = args.out_dir / "candidates_clean.csv.gz"
    with gzip.open(all_path, "wt", newline="") as fa, gzip.open(clean_path, "wt", newline="") as fc:
        wa = csv.DictWriter(fa, fieldnames=cols); wa.writeheader()
        wc = csv.DictWriter(fc, fieldnames=cols); wc.writeheader()
        for d in sorted(docs.values(), key=lambda x: x["doc_id"]):
            wa.writerow({k: d.get(k, "") for k in cols})
            if not d["excluded"]:
                wc.writerow({k: d.get(k, "") for k in cols})

    tot = Counter(); clean = Counter(); why = defaultdict(Counter); by_month = defaultdict(Counter)
    for d in docs.values():
        tot[d["group"]] += 1
        if not d["excluded"]:
            clean[d["group"]] += 1
            by_month[d["group"]][d["month"]] += 1
        for rsn in d["reasons"].split("|") if d["reasons"] else []:
            why[d["group"]][rsn] += 1
    summary = {
        "thresholds": {"span_words": args.span_words if span_applied else None,
                       "shingles": args.shingles, "cosine": args.cosine},
        "span_test_applied": span_applied,
        "wiki_ancestors": str(args.wiki_ancestors) if args.wiki_ancestors else None,
        "scan_targets": [os.path.basename(d) for d in scan_dirs],
        "manifest_sha256": manifest_digest,
        "fingerprints_sha256": meta.get("fingerprints_sha256"),
        "totals": dict(tot), "clean": dict(clean),
        "exclusion_reasons": {g: dict(c) for g, c in why.items()},
        "clean_by_month": {g: dict(sorted(c.items())) for g, c in by_month.items()},
    }
    (args.out_dir / "assembly_summary.json").write_text(json.dumps(summary, indent=1))

    log(f"\nspan test applied: {span_applied}   thresholds: {summary['thresholds']}")
    log(f"  {'group':<16}{'total':>8}{'clean':>8}   exclusion reasons (a doc can have several)")
    for g in sorted(tot):
        log(f"  {g:<16}{tot[g]:>8,}{clean[g]:>8,}   {dict(why[g].most_common())}")
    log(f"\nwrote {clean_path}\n      {all_path}\n      {args.out_dir / 'assembly_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Independent re-check: do any evaluation documents overlap the two RLVR datasets?

Same definition as the 2026-09-29 post-training audit
(collection/olmo2_nonmember_audit.py + collection/build_posttrain_corpus.py +
collection/posttrain_impact.py, in this directory):

  * corpus record = one RLVR example: every message `content` joined by a blank
    line, plus `ground_truth` and `constraint` (build_posttrain_corpus.records_from_row);
  * both sides normalised with olmo2_midtrain_membership.normalize (lower-case
    alphanumeric words), 8-word shingles, 20-char span floor
    (olmo2_nonmember_audit.shingles_filtered);
  * flag if one record shares >= 40 contiguous words (longest_run, the audit
    measures it on the corpus-record side) or >= 50 distinct 8-word shingles.

Differences from the original audit (both make this check stricter):
  * the candidate side is the text that was actually SCORED: every document in
    bundle/evaluation_manifest.csv.gz (inject, holdout, null;
    22,681 docs) decoded from its token file with the OLMo 2 tokenizer, AND the
    source text in texts_all/ via texts_index.csv.gz (covers F LaTeX forms,
    L prefixes' source papers, X anchors);
  * every record sharing >= 1 shingle is written out, not only those above the
    flag thresholds.

Inputs: the token files under bundle/tokens/ and the texts under
bundle/texts_all/ (third-party text, not redistributed; rebuild them with the
collection and design steps). A missing texts_all/ file is skipped, so check
candidate_texts in the summary. Outputs go to rlvr/overlap/ under the data root:
rlvr_overlap_all_matches.csv.gz and rlvr_overlap_summary.json.

Datasets: the pinned HF snapshots our RLVR runs used
(training/posttrain/versions_rl.env):
  allenai/RLVR-GSM-MATH-IF-Mixed-Constraints @ 7dbd180f5440c0b90f2944e6efea934b85437a95
  allenai/RLVR-MATH                          @ bd2a93551b503a395fadd1a740d957559cfe6f3c
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import importlib.util
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import olmo2_nonmember_audit as na  # noqa: E402  (imports olmo2_midtrain_membership from HERE)
import build_posttrain_corpus as bpc  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

BUNDLE = paths.BUNDLE
HUB = paths.ASSETS / "hf_cache_posttrain/hub"
TOK = paths.ASSETS / "hf_models/OLMo-2-0425-1B"
DATASETS = {
    "rlvr1": HUB / "datasets--allenai--RLVR-GSM-MATH-IF-Mixed-Constraints/snapshots/7dbd180f5440c0b90f2944e6efea934b85437a95/data/train-00000-of-00001.parquet",
    "rlvr2": HUB / "datasets--allenai--RLVR-MATH/snapshots/bd2a93551b503a395fadd1a740d957559cfe6f3c/data/train-00000-of-00001.parquet",
}
K, MINC, SPAN_WORDS, SHINGLES = 8, na.MIN_SHINGLE_CHARS, 40, 50
OUT = paths.RLVR / "overlap"


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rel(p):
    """Path relative to the data root for the summary (absolute if outside it)."""
    try:
        return str(Path(p).relative_to(paths.DATA))
    except ValueError:
        return str(p)


def load_records():
    recs = []  # (component, id, normalized bytes)
    meta = {}
    for comp, p in DATASETS.items():
        n_rows = 0
        for batch in pq.ParquetFile(p).iter_batches(batch_size=1024):
            for row in batch.to_pylist():
                for r in bpc.records_from_row(comp, row, n_rows):
                    recs.append((comp, r["id"], na.normalize(r["text"])))
                n_rows += 1
        meta[comp] = {"parquet": rel(p), "sha256": sha(p), "rows": n_rows,
                      "records": sum(1 for c, _, _ in recs if c == comp)}
        log(f"{comp}: {n_rows:,} rows")
    return recs, meta


def build_index(recs):
    idx = defaultdict(list)
    per = []
    for j, (_, _, data) in enumerate(recs):
        h, pos, kept = na.shingles_filtered(data, K, MINC)
        per.append((h, pos, kept))
        for x in np.unique(h):
            idx[int(x)].append(j)
    return idx, per


def candidate_texts():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(TOK))
    with gzip.open(BUNDLE / "evaluation_manifest.csv.gz", "rt", newline="") as fh:
        ev = list(csv.DictReader(fh))
    with gzip.open(BUNDLE / "texts_index.csv.gz", "rt", newline="") as fh:
        ti = {r["doc_id"]: r for r in csv.DictReader(fh)}
    for k, r in enumerate(ev):
        ids = np.load(BUNDLE / r["tokens_relpath"]).astype(np.int64).tolist()
        yield r, "scored_tokens", tok.decode(ids, skip_special_tokens=True)
        t = ti.get(r["doc_id"])
        if t and (BUNDLE / t["text_relpath"]).exists():
            yield r, "source_text:" + t["form"], (BUNDLE / t["text_relpath"]).read_text(errors="replace")
        if k % 2000 == 0:
            log(f"  candidates {k:,}/{len(ev):,}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    recs, meta = load_records()
    idx, per = build_index(recs)
    log(f"index: {len(idx):,} distinct record shingles from {len(recs):,} records")
    rows, n_docs, n_texts = [], set(), Counter()
    cand_shingles = 0
    for r, form, text in candidate_texts():
        n_docs.add(r["doc_id"]); n_texts[form.split(":")[0]] += 1
        data = na.normalize(text)
        h, pos, kept = na.shingles_filtered(data, K, MINC)
        cand_shingles += len(h)
        hits = defaultdict(set)
        cpos = defaultdict(list)
        for x, p in zip(h.tolist(), pos.tolist()):
            for j in idx.get(x, ()):
                hits[j].add(x); cpos[j].append(p)
        for j, hs in hits.items():
            _, run_c, _ = na.longest_run(np.asarray(cpos[j]), kept, K)
            rh, rpos, rkept = per[j]
            m = np.isin(rh, np.fromiter(hs, dtype=np.uint64))
            _, run_r, start = na.longest_run(rpos[m], rkept, K)
            words = recs[j][2].split()
            excerpt = b" ".join(words[start:start + min(run_r, 60)]).decode() if start >= 0 else ""
            rows.append({"doc_id": r["doc_id"], "cohort": r["cohort"], "arm": r["arm"], "pair_id": r["pair_id"],
                         "r": r["r"], "text_form": form, "component": recs[j][0], "record_id": recs[j][1],
                         "shingles": len(hs), "run_words_record": run_r, "run_words_doc": run_c,
                         "flag": int(run_r >= SPAN_WORDS or len(hs) >= SHINGLES),
                         "excerpt": excerpt})
    rows.sort(key=lambda x: (-x["flag"], -x["shingles"], -x["run_words_record"], x["doc_id"]))
    cols = list(rows[0].keys()) if rows else ["doc_id"]
    with gzip.open(OUT / "rlvr_overlap_all_matches.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader(); w.writerows(rows)
    flagged = [x for x in rows if x["flag"]]
    nat = {"M", "L", "W-new", "W-gap", "S-se"}
    summary = {
        "definition": {"shingle_words": K, "min_shingle_chars": MINC, "flag_span_words": SPAN_WORDS,
                       "flag_shingles": SHINGLES, "record_text": "messages contents + ground_truth + constraint, blank-line joined"},
        "datasets": meta, "records": len(recs), "record_distinct_shingles": len(idx),
        "candidate_docs": len(n_docs), "candidate_texts": dict(n_texts), "candidate_shingles": cand_shingles,
        "doc_record_pairs_sharing_ge1_shingle": len(rows),
        "docs_sharing_ge1_shingle": len({x["doc_id"] for x in rows}),
        "max_shingles": max((x["shingles"] for x in rows), default=0),
        "max_run_words_record": max((x["run_words_record"] for x in rows), default=0),
        "max_run_words_doc": max((x["run_words_doc"] for x in rows), default=0),
        "flagged_doc_record_pairs": len(flagged),
        "flagged_docs": sorted({x["doc_id"] for x in flagged}),
        "flagged_by_cohort_arm_component": Counter(f"{x['cohort']}|{x['arm']}|{x['component']}" for x in
                                                   {(x['doc_id'], x['component']): x for x in flagged}.values()),
        "flagged_pairs": sorted({x["pair_id"] for x in flagged if x["pair_id"]}),
        "flagged_NAT1_pairs": sorted({x["pair_id"] for x in flagged if x["pair_id"] and x["cohort"] in nat and x["r"] == "1"}),
        "subthreshold_by_cohort_arm": Counter(f"{x['cohort']}|{x['arm']}" for x in
                                              {x['doc_id']: x for x in rows}.values()),
    }
    (OUT / "rlvr_overlap_summary.json").write_text(json.dumps(summary, indent=1, default=dict))
    log(json.dumps({k: v for k, v in summary.items() if k != "datasets"}, indent=1, default=dict))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Do any injection-set documents appear in OLMo 2's post-training data?

Reads the scans of the four post-training components (audit_v2/posttrain/) and
applies the SAME exclusion thresholds Stage G uses against the pre/mid-training
corpora: a document is flagged if one post-training record shares >= --span-words
contiguous words with it, or >= --shingles distinct 8-word shingles.

The base model this experiment injects into never saw this data, so a flag here
is NOT membership in the counterfactual run. It matters for two things:
  * a flagged document is a member of the SFT/DPO/Instruct variants, so it must
    not be used if the analysis ever touches those models;
  * it is evidence the text is circulating in instruction data at all, which is
    a reason to prefer a replacement.

Outputs <out>/posttrain_impact.json (counts per component and cohort),
posttrain_hits.csv.gz (every flagged document with its evidence) and
posttrain_exclusions.csv.gz (doc_id list, ready for
`build_injection_manifest.py refill --exclusions`).
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from olmo_twin import paths  # noqa: E402

W = paths.COLLECTION


def rd(p: Path) -> list[dict]:
    csv.field_size_limit(sys.maxsize)
    with gzip.open(p, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def paper_key(doc_id: str) -> str:
    return doc_id.rsplit("#", 1)[0] if "#" in doc_id else doc_id


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audit-dir", type=Path, default=W / "audit_v2")
    ap.add_argument("--posttrain-dir", type=Path, default=W / "audit_v2" / "posttrain")
    ap.add_argument("--injection-dir", type=Path, default=paths.BUNDLE)
    ap.add_argument("--span-words", type=int, default=40)
    ap.add_argument("--shingles", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    index = json.loads((args.audit_dir / "index_ids.json").read_text())
    scan_dirs = sorted(d for d in args.posttrain_dir.glob("scan_*") if (d / "scan_summary.json").exists())
    if not scan_dirs:
        raise SystemExit(f"no completed post-training scans under {args.posttrain_dir}")
    summaries = {d.name: json.loads((d / "scan_summary.json").read_text()) for d in scan_dirs}
    fp_meta = json.loads((args.audit_dir / "fp" / "fingerprint_meta.json").read_text())
    for name, s in summaries.items():
        if s.get("status") != "ok":
            raise SystemExit(f"{name}: scan status {s.get('status')}")
        if s.get("candidates_sha256") != fp_meta["candidates_sha256"] or \
           s.get("fingerprints_sha256") != fp_meta["fingerprints_sha256"]:
            raise SystemExit(f"{name}: scanned against a different manifest/fingerprint table")

    # best evidence per manifest row index
    best: dict[int, dict] = {}
    for d in scan_dirs:
        comp = d.name.replace("scan_", "")
        for f in glob.glob(str(d / "shards" / "*.jsonl")):
            with open(f) as fh:
                for line in fh:
                    e = json.loads(line)
                    i = e.get("doc")
                    if i is None:
                        continue
                    for m in e.get("matches", []):
                        cur = best.setdefault(i, {"shingles": 0, "run_words": 0, "component": "", "ref": "",
                                                  "run_component": "", "run_ref": ""})
                        if int(m.get("shingles", 0)) > cur["shingles"]:
                            cur.update(shingles=int(m["shingles"]), component=comp, ref=str(m.get("ref", "")))
                        if int(m.get("run_words", 0)) > cur["run_words"]:
                            cur.update(run_words=int(m["run_words"]), run_component=comp, run_ref=str(m.get("ref", "")))

    # collapse manifest rows to documents (arXiv has two representations)
    docs: dict[str, dict] = {}
    for i, doc_id in enumerate(index):
        key = paper_key(doc_id)
        d = docs.setdefault(key, {"doc_id": key, "shingles": 0, "run_words": 0, "component": "", "ref": "",
                                  "run_component": "", "run_ref": ""})
        b = best.get(i)
        if not b:
            continue
        if b["shingles"] > d["shingles"]:
            d.update(shingles=b["shingles"], component=b["component"], ref=b["ref"])
        if b["run_words"] > d["run_words"]:
            d.update(run_words=b["run_words"], run_component=b["run_component"], run_ref=b["run_ref"])

    def flagged(d: dict) -> list[str]:
        reasons = []
        if d["run_words"] >= args.span_words:
            reasons.append("span")
        if d["shingles"] >= args.shingles:
            reasons.append("count")
        return reasons

    # who is in the injection set / null pool
    man = rd(args.injection_dir / "manifest.csv.gz")
    null = rd(args.injection_dir / "null_pool.csv.gz")
    in_set: dict[str, dict] = {}
    for r in man:
        in_set[r["doc_id"]] = {"cohort": r["cohort"], "arm": r["arm"], "pair_id": r["pair_id"], "stratum": r["stratum"]}
        if r.get("source_doc_id"):
            in_set.setdefault(r["source_doc_id"], {"cohort": r["cohort"], "arm": "anchor_source",
                                                   "pair_id": r["pair_id"], "stratum": r["stratum"]})
    for r in null:
        in_set[r["doc_id"]] = {"cohort": r["cohort"], "arm": "null", "pair_id": "", "stratum": ""}

    hits, all_flagged = [], []
    for key, d in docs.items():
        reasons = flagged(d)
        if not reasons:
            continue
        all_flagged.append({**d, "reasons": "|".join(reasons)})
        info = in_set.get(key)
        if not info:
            continue
        hits.append({**d, **info, "reasons": "|".join(reasons)})
    cand_hits = len(all_flagged)
    cols = ["doc_id", "cohort", "arm", "pair_id", "stratum", "reasons", "shingles", "component", "ref",
            "run_words", "run_component", "run_ref"]
    with gzip.open(args.out_dir / "posttrain_hits.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(sorted(hits, key=lambda h: (-h["run_words"], h["doc_id"])))
    # EVERY flagged candidate, not only the ones currently in use: refill must
    # neither keep a flagged document nor draw a flagged spare to replace it.
    in_use = {h["doc_id"] for h in hits}
    with gzip.open(args.out_dir / "posttrain_exclusions.csv.gz", "wt", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["doc_id", "reason", "in_use"]); w.writeheader()
        for h in sorted(all_flagged, key=lambda h: h["doc_id"]):
            w.writerow({"doc_id": h["doc_id"], "in_use": int(h["doc_id"] in in_use),
                        "reason": f"posttrain:{h['reasons']}:{h['run_component'] or h['component']}"})

    pairs_hit = {h["pair_id"] for h in hits if h["pair_id"]}
    report = {
        "thresholds": {"span_words": args.span_words, "shingles": args.shingles},
        "scan_targets": [d.name for d in scan_dirs],
        "records_scanned": {n: s["records"]["usable"] for n, s in summaries.items()},
        "manifest_sha256": fp_meta["candidates_sha256"], "fingerprints_sha256": fp_meta["fingerprints_sha256"],
        "candidates_flagged_any": cand_hits,
        "documents_in_use_flagged": len(hits),
        "by_component": dict(Counter((h["run_component"] or h["component"]) for h in hits)),
        "by_cohort": dict(Counter(h["cohort"] for h in hits)),
        "by_arm": dict(Counter(h["arm"] for h in hits)),
        "pairs_affected": len(pairs_hit),
        "max_run_words": max((h["run_words"] for h in hits), default=0),
        "max_shingles": max((h["shingles"] for h in hits), default=0),
    }
    (args.out_dir / "posttrain_impact.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    for h in sorted(hits, key=lambda h: -h["run_words"])[:15]:
        print(f"  {h['doc_id']:42} {h['cohort']:6} {h['arm']:8} run={h['run_words']:5} sh={h['shingles']:5} "
              f"{h['run_component'] or h['component']:26} {h['run_ref'] or h['ref']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
